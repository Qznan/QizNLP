#!/usr/bin/env python
# coding=utf-8
"""
@Author : yonas
@Time   : 2020/3/15 下午10:54
@File   : example.py
"""
import jieba
import numpy as np
import tensorflow as tf
import os, time, re, sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_dir)
sys.path.append(curr_dir + '/..')

import common.utils as utils
from model.cls_model import Model


class Deplpy_CLS_Model():
    def __init__(self, ckpt_name=None, pbmodel_dir=None):
        assert ckpt_name or pbmodel_dir, 'at least have one way'
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True),
                                     )
        self.sess = tf.Session(config=self.config, graph=self.graph)

        self.token2id_dct = {
            'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/word2id.dct', use_line_no=True),
            'label2id': utils.Any2Id.from_file(f'{curr_dir}/../data/label2id.dct', use_line_no=True),
        }
        self.jieba = jieba.Tokenizer()
        self.tokenize = lambda t: self.jieba.lcut(re.sub(r'\s+', '，', t))
        self.cut = lambda t: ' '.join(self.tokenize(t))
        if ckpt_name:
            self.load_from_ckpt_meta(ckpt_name)
        else:
            self.load_from_pbmodel(pbmodel_dir)

        self.id2label = self.token2id_dct['label2id'].get_reverse()

    def load_from_ckpt_meta(self, ckpt_name):
        self.model, self.saver = Model.from_ckpt_meta(ckpt_name, self.sess, self.graph)

    def load_from_pbmodel(self, pbmodel_dir):
        self.model = Model.from_pbmodel(pbmodel_dir, self.sess)

    def predict(self, sent, need_cut=True):
        if need_cut:
            sent = self.cut(sent)
        feed_dict = self.model.create_feed_dict_from_raw([sent], [], self.token2id_dct)
        prob = self.sess.run(self.model.y_prob, feed_dict)[0]
        pred = np.argmax(prob)  # [batch]
        return self.id2label[pred]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    dcm = Deplpy_CLS_Model(ckpt_name=curr_dir + '/../run/cls_ckpt_1/trans_mhattnpool-16-1.50-7.29-0.932-0.156.ckpt-160')
    print(dcm.predict('这个是啥'))

    dcm = Deplpy_CLS_Model(pbmodel_dir=curr_dir + '/../run/cls_pbmodel')
    print(dcm.predict('这个是啥'))
