#!/usr/bin/env python
# coding=utf-8
import jieba
import numpy as np
import tensorflow as tf
import os, time, re, sys

""" 
deploy cls_model example(for toutiao dataset)
"""
curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_dir)
sys.path.append(curr_dir + '/..')

import qiznlp.common.utils as utils
from qiznlp.model.cls_model import Model


class Deplpy_CLS_Model():
    def __init__(self, ckpt_name=None, pbmodel_dir=None):
        assert ckpt_name or pbmodel_dir, 'ues at least one way'
        self.graph = tf.Graph()
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=tf.GPUOptions(allow_growth=True),
                                     )
        self.sess = tf.Session(config=self.config, graph=self.graph)

        self.token2id_dct = {
            'word2id': utils.Any2Id.from_file(f'{curr_dir}/../data/toutiao_cls_word2id.dct', use_line_no=True),
            'label2id': utils.Any2Id.from_file(f'{curr_dir}/../data/toutiao_cls_label2id.dct', use_line_no=True),
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用CPU设为'-1'

    # option 1. from ckpt
    # assume ckpt file is 'cls_ckpt_toutiao1/trans_mhattnpool-7-0.58-0.999.ckpt-854'
    dcm = Deplpy_CLS_Model(ckpt_name=f'{curr_dir}/../run/cls_ckpt_toutiao1/trans_mhattnpool-7-0.58-0.999.ckpt-854')
    print(dcm.predict('去日本的邮轮游需要5万的资产证明吗？'))


    # option 2. from pbmodel
    # firstly export pbmodel in [cls_pbmodel_dir] with:
    """
    rm_cls = Run_Model_Cls('trans_mhattnpool')
    rm.export_model(f'{curr_dir}/../run/cls_pbmodel_dir')
    """
    # then exec follow:
    dcm = Deplpy_CLS_Model(pbmodel_dir=f'{curr_dir}/../run/cls_pbmodel_dir')
    print(dcm.predict('去日本的邮轮游需要5万的资产证明吗？'))
