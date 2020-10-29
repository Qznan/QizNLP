import os, sys, re
import time
import jieba
import numpy as np
import tensorflow as tf

import qiznlp.common.utils as utils
from qiznlp.run.run_base import Run_Model_Base, check_and_update_param_of_model_pyfile

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_dir + '/..')  # 添加上级目录即默认qiznlp根目录
from model.cls_model import Model as CLS_Model

try:
    import horovod.tensorflow as hvd
    # 示例：horovodrun -np 2 -H localhost:2 python run_cls.py
    # 可根据local_rank设置shard的数据，保证各个gpu采样的数据不重叠。
except:
    HVD_ENABLE = False
else:
    HVD_ENABLE = True

conf = utils.dict2obj({
    'early_stop_patience': None,
    'just_save_best': True,
    'n_epochs': 20,
    'data_type': 'tfrecord',
    # 'data_type': 'pkldata',
})


def print_(*args, should_print=True, **kwargs):
    if should_print:
        print(*args, **kwargs)


class Run_Model_Cls(Run_Model_Base):
    def __init__(self, model_name, tokenize=None, pbmodel_dir=None, use_hvd=False):
        # 维护sess graph config saver
        self.model_name = model_name
        if tokenize is None:
            self.jieba = jieba.Tokenizer()
            # self.jieba.load_userdict(f'{curr_dir}/segword.dct')
            self.tokenize = lambda t: self.jieba.lcut(re.sub(r'\s+', '，', t))
        else:
            self.tokenize = tokenize
        self.cut = lambda t: ' '.join(self.tokenize(t))
        self.token2id_dct = {
            # 'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/cls_word2id.dct', use_line_no=True),  # 自有数据
            # 'label2id': utils.Any2Id.from_file(f'{curr_dir}/../data/cls_label2id.dct', use_line_no=True),  # 自有数据
            'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/toutiao_cls_word2id.dct', use_line_no=True),  # toutiao新闻
            'label2id': utils.Any2Id.from_file(f'{curr_dir}/../data/toutiao_cls_label2id.dct', use_line_no=True),  # toutiao新闻
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
            self.model = CLS_Model.from_pbmodel(pbmodel_dir, self.sess)
        else:
            with self.graph.as_default():
                self.model = CLS_Model(model_name=self.model_name, run_model=self)
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

    def train(self, ckpt_dir, raw_data_file, preprocess_raw_data, batch_size=100, save_data_prefix=None):
        save_data_prefix = os.path.basename(ckpt_dir) if save_data_prefix is None else save_data_prefix
        train_epo_steps, dev_epo_steps, test_epo_steps, gen_feed_dict = self.prepare_data(conf.data_type,
                                                                                          raw_data_file,
                                                                                          preprocess_raw_data,
                                                                                          batch_size,
                                                                                          save_data_prefix=save_data_prefix,
                                                                                          update_txt=True,
                                                                                          )
        self.is_master = True
        if self.use_hvd and self.hvd_rank != 0:  # 分布式训练且非master
            dev_epo_steps, test_epo_steps = None, None  # 不进行验证和测试
            self.is_master = False

        # 字典大小自动对齐
        check_and_update_param_of_model_pyfile({
            'vocab_size': (self.model.conf.vocab_size, len(self.token2id_dct['word2id'])),
            'label_size': (self.model.conf.label_size, len(self.token2id_dct['label2id']))
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
                for i in range(epo_steps):
                    feed_dict = gen_feed_dict(i, epo, mode)
                    loss, acc, y_prob = self.eval_step(feed_dict)

                    epo_loss.append(loss)
                    epo_acc.append(acc)

                loss = np.mean(epo_loss)
                acc = np.mean(epo_acc)
                print(f'epo:{epo} {mode} loss {loss:.3f} '
                      f'{mode} acc {acc:.3f} '
                      f'elapsed {(time.time() - time0) / 60:.2f} min')
                train_info[epo][f'{mode}_loss'] = loss
                train_info[epo][f'{mode}_acc'] = acc

            info_str = f'{trn_loss:.2f}-{trn_acc:.3f}'
            # info_str += f'-{train_info[epo]["dev_loss"]:.2f}-{train_info[epo]["dev_acc"]:.3f}'
            # info_str += f'-{train_info[epo]["test_loss"]:.2f}-{train_info[epo]["test_acc"]:.3f}'

            if conf.just_save_best:
                if self.should_save(epo, train_info, 'dev_acc', greater_is_better=True):
                    self.delete_ckpt(ckpt_dir=ckpt_dir)  # 删掉已存在的
                    self.save(ckpt_dir=ckpt_dir, epo=epo, info_str=info_str)
            else:
                self.save(ckpt_dir=ckpt_dir, epo=epo, info_str=info_str)

            utils.obj2json(train_info, f'{ckpt_dir}/metrics.json')
            print('=' * 40, end='\n\n')
            if conf.early_stop_patience:
                if self.stop_training(conf.early_stop_patience, train_info, 'dev_acc'):
                    print('early stop training!')
                    print('train_info', train_info)
                    break

    def predict(self, s1_lst, need_cut=True, batch_size=100):
        if need_cut:
            s1_lst = [self.cut(s1) for s1 in s1_lst]
        if not hasattr(self, 'label2id'): self.label2id = self.token2id_dct['label2id']
        if not hasattr(self, 'id2label'): self.id2label = self.token2id_dct['label2id'].get_reverse()

        pred_lst = []
        for i in range(0, len(s1_lst), batch_size):
            batch_s1 = s1_lst[i:i + batch_size]
            feed_dict = self.model.create_feed_dict_from_raw(batch_s1, [], self.token2id_dct, mode='infer')
            probs = self.sess.run(self.model.y_prob, feed_dict)  # [batch, num_cls]
            pred = np.argmax(probs, -1)  # [batch]
            pred_label = [self.id2label[p] for p in pred]
            pred_prob = [p[pred] for p in probs]
            pred_lst.extend(zip(pred_label, pred_prob))
        return pred_lst


def preprocess_raw_data(file, tokenize, token2id_dct, **kwargs):
    """
    # 处理自有数据函数模板
    # file文件数据格式: 句子\t类别
    # [filter] 过滤
    # [segment] 分词
    # [build vocab] 构造词典
    # [split] train-dev-test
    """
    seg_file = file.rsplit('.', 1)[0] + '_seg.txt'
    if not os.path.exists(seg_file):
        items = utils.file2items(file)
        # 过滤
        # filter here

        print('过滤后数据量', len(items))

        # 分词
        for i, item in enumerate(items):
            item[0] = ' '.join(tokenize(item[0]))
        utils.list2file(seg_file, items)
        print('保存分词后数据成功', '数据量', len(items), seg_file)
    else:
        # 读取分词好的数据
        items = utils.file2items(seg_file)

    # 划分
    train_items, dev_items, test_items = utils.split_file(items, ratio='18:1:1', shuffle=True, seed=1234)

    # 构造词典(option)
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
                    token2id_dct['word2id'].to_count(item[0].split(' '))
                if 'label2id' in need_to_rebuild:
                    token2id_dct['label2id'].to_count([item[1]])
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=5, max_vocab_size=30000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/cls_word2id.dct')
        if 'label2id' in need_to_rebuild:
            token2id_dct['label2id'].rebuild_by_counter(restrict=['<unk>'])
            token2id_dct['label2id'].save(f'{curr_dir}/../data/cls_label2id.dct')
    else:
        print('使用已有词表文件...')

    return train_items, dev_items, test_items


def preprocess_common_dataset_Toutiao(file, tokenize, token2id_dct, **kwargs):
    train_file = f'{curr_dir}/../data/train.toutiao.cls.txt'
    dev_file = f'{curr_dir}/../data/valid.toutiao.cls.txt'
    test_file = f'{curr_dir}/../data/test.toutiao.cls.txt'
    items_lst = []
    for file in [train_file, dev_file, test_file]:
        seg_file = file.rsplit('.', 1)[0] + '_seg.txt'  # 原始文本分词并保存为_seg.txt后缀文件
        if not os.path.exists(seg_file):
            items = utils.file2items(file, deli='\t')
            # 过滤
            # filter here

            print('过滤后数据量', len(items))

            # 分词
            for i, item in enumerate(items):
                item[0] = ' '.join(tokenize(item[0]))
            utils.list2file(seg_file, items)
            print('保存分词后数据成功', '数据量', len(items), seg_file)
            items_lst.append(items)
        else:
            # 读取分词好的数据
            items_lst.append(utils.file2items(seg_file))

    train_items, dev_items, test_items = items_lst

    # 构造词典(option)
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
                    token2id_dct['word2id'].to_count(item[0].split(' '))
                if 'label2id' in need_to_rebuild:
                    token2id_dct['label2id'].to_count([item[1]])
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>'], min_freq=1, max_vocab_size=20000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/toutiao_cls_word2id.dct')
        if 'label2id' in need_to_rebuild:
            token2id_dct['label2id'].rebuild_by_counter(restrict=['<unk>'])
            token2id_dct['label2id'].save(f'{curr_dir}/../data/toutiao_cls_label2id.dct')
    else:
        print('使用已有词表文件...')

    return train_items, dev_items, test_items


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用CPU设为'-1'

    # rm_cls = Run_Model_Cls('trans_meanpool')  # use transformer_encoder with mean_pooling
    rm_cls = Run_Model_Cls('trans_mhattnpool')  # use transformer_encoder with multi_head_pooling
    # rm_cls = Run_Model_Cls('bert_cls')  # use transformer_encoder with multi_head_pooling

    # 训练自有数据
    # rm_cls.train('cls_ckpt_1', '../data/cls_example_data.txt', preprocess_raw_data=preprocess_raw_data, batch_size=512)  # train

    # 训练toutiao新闻语料
    rm_cls.train('cls_ckpt_toutiao1', '', preprocess_raw_data=preprocess_common_dataset_Toutiao, batch_size=128)  # train

    # exit(0)
    # demo头条新闻分类
    rm_cls.restore('cls_ckpt_toutiao1')  # for infer
    import readline
    while True:
        try:
            sent = input('enter:')
            need_cut = False if ' ' in sent else True
            time0 = time.time()
            ret = rm_cls.predict([sent], need_cut=need_cut)
            print(ret[0])
            print('elapsed:', time.time() - time0)
        except KeyboardInterrupt:
            exit(0)

