import os, sys, re
import time
import jieba
import numpy as np
import tensorflow as tf

import qiznlp.common.utils as utils
import qiznlp.common.train_helper as train_helper
from qiznlp.run.run_base import Run_Model_Base, check_and_update_param_of_model_pyfile

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_dir + '/..')  # 添加上级目录即默认qiznlp根目录
from model.multi_mch_model import Model as MMCH_Model

try:
    import horovod.tensorflow as hvd
    # 示例：horovodrun -np 2 -H localhost:2 python run_s2s.py
    # 可根据local_rank设置shard的数据，保证各个gpu采样的数据不重叠。
except:
    HVD_ENABLE = False
else:
    HVD_ENABLE = True

conf = utils.dict2obj({
    'early_stop_patience': None,
    'just_save_best': True,
    'n_epochs': 10,
    'data_type': 'tfrecord',
    # 'data_type': 'pkldata',
})


class Run_Model_MMCH(Run_Model_Base):
    def __init__(self, model_name, tokenize=None, pbmodel_dir=None, use_hvd=False):
        # 维护sess graph config saver
        self.model_name = model_name
        if tokenize is None:
            self.jieba = jieba.Tokenizer()
            # self.jieba.load_userdict(f'{curr_dir}/../data/segword.dct')
            self.tokenize = lambda t: self.jieba.lcut(re.sub(r'\s+', '，', t))
        else:
            self.tokenize = tokenize
        self.cut = lambda t: ' '.join(self.tokenize(t))
        self.token2id_dct = {
            # 'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/mmch_word2id.dct', use_line_no=True),  # 自有数据
            # 'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/mmch_char2id.dct', use_line_no=True),  # 自有数据
            'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/DB_mmch_word2id.dct', use_line_no=True),  # 豆瓣多轮语料
            'char2id': utils.Any2Id.from_file(f'{curr_dir}/../data/DB_mmch_char2id.dct', use_line_no=True),  # 豆瓣多轮语料
        }
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True),
                                     )
        self.use_hvd = use_hvd if HVD_ENABLE else False
        if self.use_hvd:
            hvd.init()
            self.hvd_rank = hvd.rank()
            self.hvd_size = hvd.size()
            self.config.gpu_options.visible_device_list = str(hvd.local_rank())
        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.config, graph=self.graph)

        if pbmodel_dir is not None:  # 只能做predict
            self.model = MMCH_Model.from_pbmodel(pbmodel_dir, self.sess)
        else:
            with self.graph.as_default():
                self.model = MMCH_Model(model_name=self.model_name, run_model=self)
                if self.use_hvd:
                    self.model.optimizer._lr = self.model.optimizer._lr * self.hvd_size  # 分布式训练大batch增大学习率
                    self.model.hvd_optimizer = hvd.DistributedOptimizer(self.model.optimizer)
                    self.model.train_op = self.model.hvd_optimizer.minimize(self.model.loss, global_step=self.model.global_step)
                self.sess.run(tf.global_variables_initializer())
                if self.use_hvd:
                    self.sess.run(hvd.broadcast_global_variables(0))

        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=100)  # must in the graph context

    def train_step(self, feed_dict):
        _, step, loss, accuracy = self.sess.run([self.model.train_op,
                                                 self.model.global_step,
                                                 self.model.loss,
                                                 self.model.accuracy,
                                                 ],
                                                feed_dict=feed_dict)
        return step, loss, accuracy

    def eval_step(self, feed_dict):
        loss, accuracy, y_prob = self.sess.run([self.model.loss,
                                                self.model.accuracy,
                                                self.model.y_prob,
                                                ],
                                               feed_dict=feed_dict)
        return loss, accuracy, y_prob

    def train(self, ckpt_dir, raw_data_file, preprocess_raw_data, batch_size = 100, save_data_prefix = None):
        save_data_prefix = os.path.basename(ckpt_dir) if save_data_prefix is None else save_data_prefix
        train_epo_steps, dev_epo_steps, test_epo_steps, gen_feed_dict = self.prepare_data(conf.data_type,
                                                                                          raw_data_file,
                                                                                          preprocess_raw_data,
                                                                                          batch_size,
                                                                                          save_data_prefix=save_data_prefix,
                                                                                          update_txt=False,
                                                                                          )
        self.is_master = True
        if hasattr(self, 'hvd_rank') and self.hvd_rank != 0:  # 分布式训练且非master
            dev_epo_steps, test_epo_steps = None, None  # 不进行验证和测试
            self.is_master = False

        # 字典大小自动对齐
        check_and_update_param_of_model_pyfile({
            'vocab_size': (self.model.conf.vocab_size, len(self.token2id_dct['word2id'])),
            'char_vocab_size': (self.model.conf.char_vocab_size, len(self.token2id_dct['char2id'])),
        }, self.model)

        train_info = {}
        for epo in range(1, 1 + conf.n_epochs):
            train_info[epo] = {}

            # train
            time0 = time.time()
            epo_num_example = 0
            trn_epo_loss = []
            trn_epo_acc = []
            for i in range(train_epo_steps):
                feed_dict = gen_feed_dict(i, epo, 'train')
                epo_num_example += feed_dict.pop('num')

                step_start_time = time.time()
                step, loss, acc = self.train_step(feed_dict)
                trn_epo_loss.append(loss)
                trn_epo_acc.append(acc)

                if self.is_master:
                    print(f'\repo:{epo} step:{i + 1}/{train_epo_steps} num:{epo_num_example} '
                          f'cur_loss:{loss:.3f} epo_loss:{np.mean(trn_epo_loss):.3f} '
                          f'epo_acc:{np.mean(trn_epo_acc):.3f} '
                          f'sec/step:{time.time() - step_start_time:.2f}',
                          end=f'{os.linesep if i == train_epo_steps - 1 else ""}',
                          )

            trn_loss = np.mean(trn_epo_loss)
            trn_acc = np.mean(trn_epo_acc)
            if self.is_master:
                print(f'epo:{epo} trn loss {trn_loss:.3f} '
                      f'trn acc {trn_acc:.3f} '
                      f'elapsed {(time.time() - time0) / 60:.2f} min')
            train_info[epo]['trn_loss'] = trn_loss
            train_info[epo]['trn_acc'] = trn_acc

            if not self.is_master:
                continue

            # dev or test
            for mode in ['dev', 'test']:
                epo_steps = {'dev': dev_epo_steps, 'test': test_epo_steps}[mode]
                if epo_steps is None:
                    continue
                time0 = time.time()
                epo_loss = []
                epo_acc = []
                # to calc recall
                epo_s1 = []
                epo_prob = []
                epo_y = []
                for i in range(epo_steps):
                    feed_dict = gen_feed_dict(i, epo, mode)
                    loss, acc, y_prob = self.eval_step(feed_dict)

                    epo_loss.append(loss)
                    epo_acc.append(acc)

                    # to calc recall
                    epo_s1.extend(feed_dict[self.model.multi_s1])
                    epo_prob.extend(y_prob)
                    epo_y.extend(feed_dict[self.model.target])

                loss = np.mean(epo_loss)
                acc = np.mean(epo_acc)
                recall = train_helper.calc_recall(epo_s1, epo_prob, epo_y, strip_pad=True)  # calc recall
                recall = list(map(lambda e: round(e, 3), recall))

                print(f'epo:{epo} {mode} loss {loss:.3f} '
                      f'{mode} acc {acc:.3f} '
                      f'{mode} recall@n {recall} '  # print recall
                      f'elapsed {(time.time() - time0) / 60:.2f} min')
                train_info[epo][f'{mode}_loss'] = loss
                train_info[epo][f'{mode}_acc'] = acc
                train_info[epo][f'{mode}_recall@n'] = recall

            info_str = f'{trn_loss:.2f}-{train_info[epo]["dev_loss"]:.2f}'
            info_str += f'-{trn_acc:.3f}-{train_info[epo]["dev_acc"]:.3f}'
            info_str += f'-{train_info[epo]["dev_recall@n"][0]:.3f}'

            if conf.just_save_best:
                if self.should_save(epo, train_info, 'dev_recall@n', greater_is_better=True):
                    self.delete_ckpt(ckpt_dir=ckpt_dir)  # 删掉已存在的
                    self.save(ckpt_dir=ckpt_dir, epo=epo, info_str=info_str)
            else:
                self.save(ckpt_dir=ckpt_dir, epo=epo, info_str=info_str)

            utils.obj2json(train_info, f'{ckpt_dir}/metrics.json')
            print('=' * 40, end='\n')
            if conf.early_stop_patience:
                if self.stop_training(conf.early_stop_patience, train_info, 'dev_loss', greater_is_better=False):
                    print('early stop training!')
                    print('train_info', train_info)
                    break

    def predict(self, multi_s1_lst, s2_lst, need_cut=True, batch_size=100):
        assert len(multi_s1_lst) == len(s2_lst)
        if need_cut:
            multi_s1_lst = ['$$$'.join([self.cut(s1) for s1 in multi_s1.split('$$$')]) for multi_s1 in multi_s1_lst]
            s2_lst = [self.cut(s2) for s2 in s2_lst]

        pred_lst = []
        for i in range(0, len(multi_s1_lst), batch_size):
            batch_multi_s1 = multi_s1_lst[i:i + batch_size]
            batch_s2 = s2_lst[i:i + batch_size]
            feed_dict = self.model.create_feed_dict_from_raw(batch_multi_s1, batch_s2, [], self.token2id_dct, mode='infer')
            probs = self.sess.run(self.model.y_prob, feed_dict)  # [batch]
            threshold = 2.5 if self.model_name.startswith('MRFN') else 0.5
            preds = [1 if prob >= threshold else 0 for prob in probs]
            pred_lst.extend(preds)
        return pred_lst


def preprocess_raw_data(file, tokenize, token2id_dct, **kwargs):
    """
    # 处理自有数据函数模板
    # file文件数据格式: 多轮对话句子1\t多轮对话句子2\t...\t多轮对话句子n
    # [filter] 过滤
    # [segment] 分词
    # [build vocab] 构造词典
    # [split] train-dev-test
    """
    seg_file = file.rsplit('.', 1)[0] + '_seg.txt'
    if not os.path.exists(seg_file):
        sess_lst = utils.file2items(file)
        # 过滤
        # filter here

        print('过滤后数据量', len(sess_lst))

        # 分词
        for i, sess in enumerate(sess_lst):
            # sess_lst[i] = [' '.join(s) for s in sess]  # 按字分
            sess_lst[i] = [' '.join(tokenize(s)) for s in sess]  # 按词分
        utils.list2file(seg_file, sess_lst)
        print('保存分词后数据成功', '数据量', len(sess_lst), seg_file)
    else:
        # 读取分词好的数据
        sess_lst = utils.file2items(seg_file)

    # 转为多轮格式 multi-turn之间用$$$分隔
    items = []
    for sess in sess_lst:
        for i in range(1, len(sess)):
            multi_src = '$$$'.join(sess[:i])
            tgt = sess[i]
            items.append([multi_src, tgt])
    # items: [['w w w$$$w w', 'w w w'],...]

    # 划分 不分测试集
    train_items, dev_items = utils.split_file(items, ratio='19:1', shuffle=True, seed=1234)

    # 构造词典(option) 字词联合
    need_to_rebuild = []
    for token2id_name in token2id_dct:
        if not token2id_dct[token2id_name]:
            print(f'字典{token2id_name} 载入不成功, 将生成并保存')
            need_to_rebuild.append(token2id_name)

    if need_to_rebuild:
        print(f'生成缺失词表文件...{need_to_rebuild}')
        for items in [train_items, dev_items]:  # 字典只统计train和dev
            for item in items:
                if 'word2id' in need_to_rebuild:
                    for sent in item[0].split('$$$'):
                        token2id_dct['word2id'].to_count(sent.split(' '))
                    token2id_dct['word2id'].to_count(item[1].split(' '))
                if 'char2id' in need_to_rebuild:
                    for sent in item[0].split('$$$'):
                        token2id_dct['char2id'].to_count(list(sent.replace(' ', '')))
                    token2id_dct['char2id'].to_count(list(item[1].replace(' ', '')))
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=1, max_vocab_size=30000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/mmch_word2id.dct')
        if 'char2id' in need_to_rebuild:
            token2id_dct['char2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=1, max_vocab_size=4000)
            token2id_dct['char2id'].save(f'{curr_dir}/../data/mmch_char2id.dct')
    else:
        print('使用已有词表文件...')

    # 负采样
    train_items = train_helper.gen_pos_neg_sample(train_items, sample_idx=1, num_neg_exm=4)
    dev_items = train_helper.gen_pos_neg_sample(dev_items, sample_idx=1, num_neg_exm=4)
    return train_items, dev_items, None


def preprocess_common_dataset_Douban(file, tokenize, token2id_dct, **kwargs):
    # 豆瓣多轮语料
    # 对话数据分词,且生成词和字的字典word2id char2id
    file = f'{curr_dir}/../data/Douban_Sess662.txt'

    def Doubanchange2items(file):
        # 转为[multi_src, tgt]格式
        # 分词
        seg_file = file.rsplit('.', 1)[0] + '_seg.txt'
        if not os.path.exists(seg_file):
            items = utils.file2items(file)
            # 分词
            for i, item in enumerate(items):
                for j in range(len(item)):
                    items[i][j] = ' '.join(tokenize(items[i][j]))
            utils.list2file(seg_file, items)
            print('保存分词后数据成功', '数据量', len(items), seg_file)
        else:
            items = utils.file2items(seg_file)

        exm_lst = []
        sess_lst = items
        for sess in sess_lst:
            for i in range(1, len(sess)):
                multi_src = '$$$'.join(sess[:i])
                tgt = sess[i]
                exm_lst.append([multi_src, tgt])
        return exm_lst

    items = Doubanchange2items(file)  # [['w w w$$$w w', 'w w w'],...]

    # 划分 不分测试集
    train_items, dev_items = utils.split_file(items, ratio='19:1', shuffle=True, seed=1234)

    # 构造词典(option) 字词联合
    need_to_rebuild = []
    for token2id_name in token2id_dct:
        if not token2id_dct[token2id_name]:
            print(f'字典{token2id_name} 载入不成功, 将生成并保存')
            need_to_rebuild.append(token2id_name)

    if need_to_rebuild:
        print(f'生成缺失词表文件...{need_to_rebuild}')
        for items in [train_items, dev_items]:  # 字典只统计train和dev
            for item in items:
                if 'word2id' in need_to_rebuild:
                    for sent in item[0].split('$$$'):
                        token2id_dct['word2id'].to_count(sent.split(' '))
                    token2id_dct['word2id'].to_count(item[1].split(' '))
                if 'char2id' in need_to_rebuild:
                    for sent in item[0].split('$$$'):
                        token2id_dct['char2id'].to_count(list(sent.replace(' ', '')))
                    token2id_dct['char2id'].to_count(list(item[1].replace(' ', '')))
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=1, max_vocab_size=30000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/DB_mmch_word2id.dct')
        if 'char2id' in need_to_rebuild:
            token2id_dct['char2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=1, max_vocab_size=4000)
            token2id_dct['char2id'].save(f'{curr_dir}/../data/DB_mmch_char2id.dct')
    else:
        print('使用已有词表文件...')

    # 负采样
    train_items = train_helper.gen_pos_neg_sample(train_items, sample_idx=1, num_neg_exm=4)
    dev_items = train_helper.gen_pos_neg_sample(dev_items, sample_idx=1, num_neg_exm=4)
    return train_items, dev_items, None


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用CPU设为'-1'

    # rm_mmch = Run_Model_MMCH('DAM')  # use DAM
    rm_mmch = Run_Model_MMCH('MRFN')  # use MRFN

    # 训练自有数据
    # rm_mmch.train('multi_mch_ckpt_1', '../data/multi_mch_example_data.txt', preprocess_raw_data=preprocess_raw_data, batch_size=512)  # train

    # 训练豆瓣多轮语料
    rm_mmch.train('multi_mch_ckpt_DB1', '', preprocess_raw_data=preprocess_common_dataset_Douban, batch_size=128)  # train

    # demo豆瓣多轮检索式对话模型
    rm_mmch.restore('multi_mch_ckpt_DB1')  # for infer
    import readline
    while True:
        try:
            inp = input('enter:($$$分隔多轮句子,|||分隔问句与答句)')
            sent1, sent2 = inp.split('|||')
            need_cut = False if ' ' in sent1 else True
            time0 = time.time()
            ret = rm_mmch.predict([sent1], [sent2], need_cut=need_cut)
            print(ret[0])
            print('elapsed:', time.time() - time0)
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            print(e)
