import os, sys, re
import time
import jieba
import numpy as np
import tensorflow as tf

import qiznlp.common.utils as utils
from qiznlp.run.run_base import Run_Model_Base, check_and_update_param_of_model_pyfile

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_dir + '/..')  # 添加上级目录即默认qiznlp根目录
from model.s2s_model import Model as S2S_Model

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
    'n_epochs': 5,
    'data_type': 'tfrecord',
    # 'data_type': 'pkldata',
})


class Run_Model_S2S(Run_Model_Base):
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
            # 'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/s2s_char2id.dct', use_line_no=True),  # 自有数据
            'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/XHJ_s2s_char2id.dct', use_line_no=True),  # 小黄鸡
            # 'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/LCCC_s2s_char2id.dct', use_line_no=True),  # LCCC
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
            self.model = S2S_Model.from_pbmodel(pbmodel_dir, self.sess)
        else:
            with self.graph.as_default():
                self.model = S2S_Model(model_name=self.model_name, run_model=self)
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
        _, step, loss = self.sess.run([self.model.train_op,
                                       self.model.global_step,
                                       self.model.loss,
                                       ],
                                      feed_dict=feed_dict)
        return step, loss

    def eval_step(self, feed_dict):
        loss, = self.sess.run([self.model.loss,
                               ],
                              feed_dict=feed_dict)
        return loss

    def train(self, ckpt_dir, raw_data_file, preprocess_raw_data, batch_size=100, save_data_prefix=None):
        save_data_prefix = os.path.basename(ckpt_dir) if save_data_prefix is None else save_data_prefix
        train_epo_steps, dev_epo_steps, test_epo_steps, gen_feed_dict = self.prepare_data(conf.data_type,
                                                                                          raw_data_file,
                                                                                          preprocess_raw_data,
                                                                                          batch_size,
                                                                                          save_data_prefix=save_data_prefix,
                                                                                          update_txt=False
                                                                                          )
        self.is_master = True
        if hasattr(self, 'hvd_rank') and self.hvd_rank != 0:  # 分布式训练且非master
            dev_epo_steps, test_epo_steps = None, None  # 不进行验证和测试
            self.is_master = False

        # 字典大小自动对齐
        check_and_update_param_of_model_pyfile({
            'vocab_size': (self.model.conf.vocab_size, len(self.token2id_dct['word2id'])),
        }, self.model)

        train_info = {}
        for epo in range(1, 1 + conf.n_epochs):
            train_info[epo] = {}

            # train
            time0 = time.time()
            epo_num_example = 0
            trn_epo_loss = []
            for i in range(train_epo_steps):
                feed_dict = gen_feed_dict(i, epo, 'train')
                epo_num_example += feed_dict.pop('num')

                step_start_time = time.time()
                step, loss = self.train_step(feed_dict)
                trn_epo_loss.append(loss)

                if self.is_master:
                    print(f'\repo:{epo} step:{i + 1}/{train_epo_steps} num:{epo_num_example} '
                          f'cur_loss:{loss:.3f} epo_loss:{np.mean(trn_epo_loss):.3f} '
                          f'sec/step:{time.time() - step_start_time:.2f}',
                          end=f'{os.linesep if i == train_epo_steps - 1 else ""}',
                          )

            trn_loss = np.mean(trn_epo_loss)
            if self.is_master:
                print(f'epo:{epo} trn loss {trn_loss:.3f} '
                      f'elapsed {(time.time() - time0) / 60:.2f} min')
            train_info[epo]['trn_loss'] = trn_loss

            if not self.is_master:
                continue

            # dev or test
            for mode in ['dev', 'test']:
                epo_steps = {'dev': dev_epo_steps, 'test': test_epo_steps}[mode]
                if epo_steps is None:
                    continue
                time0 = time.time()
                epo_loss = []
                for i in range(epo_steps):
                    feed_dict = gen_feed_dict(i, epo, mode)
                    loss = self.eval_step(feed_dict)

                    epo_loss.append(loss)

                loss = np.mean(epo_loss)
                print(f'epo:{epo} {mode} loss {loss:.3f} '
                      f'elapsed {(time.time() - time0) / 60:.2f} min')
                train_info[epo][f'{mode}_loss'] = loss

            info_str = f'{trn_loss:.2f}'
            info_str += f'-{train_info[epo]["dev_loss"]:.2f}'

            if conf.just_save_best:
                if self.should_save(epo, train_info, 'dev_loss', greater_is_better=False):
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

    def predict(self, s1_lst, need_cut=True, batch_size=100):
        if need_cut:
            s1_lst = [self.cut(s1) for s1 in s1_lst]
        if not hasattr(self, 'word2id'): self.word2id = self.token2id_dct['word2id']
        if not hasattr(self, 'id2word'): self.id2word = self.token2id_dct['word2id'].get_reverse()
        pred_s2_lst = []
        for i in range(0, len(s1_lst), batch_size):
            batch_s1 = s1_lst[i:i + batch_size]
            feed_dict = self.model.create_feed_dict_from_raw(batch_s1, [], self.token2id_dct, mode='infer')
            s2_ids, s2_score = self.sess.run([self.model.decoded_ids, self.model.scores], feed_dict)
            # s2_ids: [batch, beam, len]
            # s2_score: [batch, beam]
            s2_ids = s2_ids.tolist()
            for batch_idx in range(len(s2_ids)):
                beam_sents = s2_ids[batch_idx]
                for beam_idx, sent in enumerate(beam_sents):
                    while sent[-1] == self.word2id['<pad>'] or sent[-1] == self.word2id['<eos>']:
                        sent.pop(-1)
                    beam_sents[beam_idx] = ''.join([self.id2word.get(wid, '<unk>') for wid in sent])
                pred_s2_lst.append(beam_sents)
        return pred_s2_lst, s2_score


def preprocess_raw_data(file, tokenize, token2id_dct, **kwargs):
    """
    # 处理自有数据函数模板
    # file文件数据格式: 句子1\t句子2
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
            item[1] = ' '.join(tokenize(item[1]))
        utils.list2file(seg_file, items)
        print('保存分词后数据成功', '数据量', len(items), seg_file)
    else:
        # 读取分词好的数据
        items = utils.file2items(seg_file)

    # 划分 不分测试集
    train_items, dev_items = utils.split_file(items, ratio='19:1', shuffle=True, seed=1234)

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
                    token2id_dct['word2id'].to_count(item[1].split(' '))
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=5, max_vocab_size=30000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/s2s_word2id.dct')
    else:
        print('使用已有词表文件...')


    return train_items, dev_items, None


def preprocess_common_dataset_XiaoHJ(file, tokenize, token2id_dct, **kwargs):
    # 小黄鸡语料（直接按字分）
    file = f'{curr_dir}/../data/XHJ_5w.txt'

    def change2items(file):
        # 转为[src, tgt]格式 且按字分
        lines = utils.file2list(file)
        items = [line.split(' ', 1) for line in lines]
        exm_lst = []
        sent_lst = []
        for item in items:
            if len(item) == 1 and item[0] == 'E':  # 分隔标志
                if sent_lst:
                    src_tgt_lst = zip(sent_lst, sent_lst[1:])
                    exm_lst.extend([[' '.join(src), ' '.join(tgt)] for src, tgt in src_tgt_lst])
                    sent_lst = []
                continue
            if item[0] == 'M' and item[1]:  # 有些数据只有M
                sent_lst.append(item[1])
        if sent_lst:
            src_tgt_lst = zip(sent_lst, sent_lst[1:])
            exm_lst.extend([[' '.join(src), ' '.join(tgt)] for src, tgt in src_tgt_lst])
        return exm_lst

    # XiaoHJ数据不分词,直接按字分,但为了方便词典仍旧叫word2id
    items = change2items(file)

    # 划分 不分测试集
    train_items, dev_items = utils.split_file(items, ratio='19:1', shuffle=True, seed=1234)

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
                    token2id_dct['word2id'].to_count(item[0].split(' '))  # 按字分
                    token2id_dct['word2id'].to_count(item[1].split(' '))  # 按字分
        if 'word2id' in need_to_rebuild:
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=1, max_vocab_size=4000)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/XHJ_s2s_char2id.dct')
    else:
        print('使用已有词表文件...')
    return train_items, dev_items, None


def preprocess_common_dataset_LCCC(file, tokenize, token2id_dct, **kwargs):
    # LCCC语料（已分词）
    import json
    # please download LCCC corpus (https://github.com/thu-coai/CDial-GPT)
    # by yourself and save in data dir like follow:
    train_file = f'{curr_dir}/../data/LCCC-base-split/LCCC-base_train.json'
    dev_file = f'{curr_dir}/../data/LCCC-base-split/LCCC-base_valid.json'
    test_file = f'{curr_dir}/../data/LCCC-base-split/LCCC-base_test.json'

    def change2items(file):
        corpus_data = json.load(open(file, 'r', encoding='U8'))
        exm_lst = []
        # transform multi-turn session to single-turn example
        for sess in corpus_data:  # sess: sent list
            src_tgt_lst = zip(sess, sess[1:])
            for src, tgt in src_tgt_lst:
                src = ' '.join(src.replace(' ', ''))  # 按字分
                tgt = ' '.join(tgt.replace(' ', ''))  # 按字分
                exm_lst.append([src, tgt])  # already segment
        return exm_lst

    train_items = change2items(train_file)
    dev_items = change2items(dev_file)
    test_items = change2items(test_file)

    # 构造词典(option)
    need_to_rebuild = []
    for name, any2id in token2id_dct.items():
        if not any2id:  # len(any2id) = 0
            print(f'字典{name} 载入不成功, 将生成并保存')
            need_to_rebuild.append(name)

    if need_to_rebuild:
        print(f'生成词表文件...{need_to_rebuild}')
        for items in [train_items, dev_items]:  # 字典只统计train,dev
            for item in items:
                if 'word2id' in need_to_rebuild:
                    token2id_dct['word2id'].to_count(item[0].split(' '))
                    token2id_dct['word2id'].to_count(item[1].split(' '))
        if 'word2id' in need_to_rebuild:
            # token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=100, max_vocab_size=40000)
            # token2id_dct['word2id'].save(f'{curr_dir}/../data/LCCC_s2s_word2id.dct')
            token2id_dct['word2id'].rebuild_by_counter(restrict=['<pad>', '<unk>', '<eos>'], min_freq=10, max_vocab_size=4500)
            token2id_dct['word2id'].save(f'{curr_dir}/../data/LCCC_s2s_char2id.dct')

    else:
        print('使用已有词表文件...')
    return train_items, dev_items, test_items


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用CPU设为'-1'

    rm_s2s = Run_Model_S2S('trans')  # use transformer seq2seq
    # rm_s2s = Run_Model_S2S('rnn_s2s')  # use biGRU encoder + bah_attn + GRU decoder

    # 训练自有数据
    # rm_s2s.train('s2s_ckpt_1', '../data/s2s_example_data.txt', preprocess_raw_data=preprocess_raw_data, batch_size=512)  # train

    # 训练小黄鸡
    rm_s2s.train('s2s_ckpt_XHJ1', '', preprocess_raw_data=preprocess_common_dataset_XiaoHJ, batch_size=512)  # train

    # 训练LCCC语料
    # rm_s2s.train('s2s_ckpt_LCCC1', '', preprocess_raw_data=preprocess_common_dataset_LCCC, batch_size=512)  # train

    # exit(0)
    # demo小黄鸡聊天机器人
    rm_s2s.restore('s2s_ckpt_XHJ1')  # for infer
    import readline

    from qiznlp.common.modules.rerank import rerank
    while True:
        try:
            inp = input('enter:')
            sent1 = ' '.join(inp)  # 小黄鸡分字
            time0 = time.time()
            batch_sent, batch_score = rm_s2s.predict([sent1], need_cut=False)
            print(*[f'{sent_}\t{score_}' for sent_, score_, in zip(batch_sent[0], batch_score[0])], sep='\n')
            print('elapsed:', time.time() - time0)

            # use rerank module
            time0 = time.time()
            ori_results = [[sent_, score_] for sent_, score_, in zip(batch_sent[0], batch_score[0])]
            rarank_results = rerank(ori_results, input_sent=inp)
            print('after rerank >>>>>>>>>>>>>>:')
            print(*[f'{sent_}\t{score_}' for sent_, score_, in rarank_results], sep='\n')
            print('elapsed:', time.time() - time0)

        except KeyboardInterrupt:
            exit(0)
