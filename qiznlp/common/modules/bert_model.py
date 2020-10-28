#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
from .bert import modeling
from .bert import tokenization

class BERT(object):
    def __init__(self,
                 bert_model_dir,
                 is_training,
                 input_ids,  # [batch,len]
                 input_mask=None,  # [batch,len]
                 segment_ids=None,  # [batch,len]
                 verbose=False
                 ):
        self.bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_model_dir, 'bert_config.json'))
        self.is_training = is_training
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.bm = None
        self.build_model(self.input_ids, self.input_mask, self.segment_ids)
        if self.is_training:
            self.restore_vars(os.path.join(bert_model_dir, 'bert_model.ckpt'), verbose=verbose)

    def build_model(self, input_ids, input_mask=None, segment_ids=None):
        self.bm = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )

    def get_pooled_output(self):
        """ 获取对应的bert的[CLS]输出[batch_size, hidden_size] """
        if not self.bm:
            print('bert model has not build yet, please call build_model()')
        return self.bm.get_pooled_output()

    def get_sequence_output(self):
        """ 获取对应的bert输出[batch_size, seq_length, hidden_size] """
        if not self.bm:
            print('bert model has not build yet, please call build_model()')
        return self.bm.get_sequence_output()

    def get_seqlen(self):
        """ 获取input_ids真实长度 """
        seqlen = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.input_ids)), axis=1), tf.int32)  # [batch]
        return seqlen

    def get_nonpad_mask(self):
        """ 获取input_ids的mask: 1 for nonpad(!=0) - 0 for pad(=0) """
        mask = tf.sign(tf.abs(self.input_ids))  # [batch,len]
        return mask

    def restore_vars(self, init_checkpoint, verbose=False):
        """ 加载预训练的BERT模型参数 """
        print(f'<<<<<< restoring bert vars from ckpt: {init_checkpoint}')
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if verbose:
            print("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print(f'  name = {var.name}, shape = {var.shape}{init_string}')

def get_tokenizer(vocab_file):
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)