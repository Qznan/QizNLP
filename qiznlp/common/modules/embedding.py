#!/usr/bin/env python
# coding=utf-8
from .common_layers import *


def embedding(ids, vocab_size, embedding_size, name='embedding', reuse=tf.AUTO_REUSE, pad_id=0, scale_sqrt_depth=True,
              pretrain_embedding=None, pretrain_trainable=True, word_dropout_rate=0.):
    """ embedding """
    # ids 3-D Tensor [batch, length, 1]
    if pretrain_embedding is None:
        with tf.variable_scope(name, reuse=reuse):
            var = tf.get_variable('weights', [vocab_size, embedding_size],  # [vocab,embed]
                                  initializer=tf.random_normal_initializer(0.0, embedding_size ** -0.5))
    else:
        with tf.variable_scope(name, reuse=reuse):
            var = tf.get_variable('weights', [vocab_size, embedding_size],  # [vocab,embed]
                                  trainable=pretrain_trainable,
                                  initializer=tf.constant_initializer(pretrain_embedding, dtype=tf.float32))

    # word level drop out
    if word_dropout_rate:
        ids = dropout_no_scaling(ids, 1.0 - word_dropout_rate)  # 随机将部分id变为0,相当于将单词变为pad

    # lookup table
    embedding = tf.gather(var, ids)  # [batch,length,1,hidden]
    embedding = tf.squeeze(embedding, axis=-2)  # [batch,length,hidden]
    if scale_sqrt_depth:
        embedding *= embedding_size ** 0.5
    embedding = embedding * tf.to_float(tf.not_equal(ids, pad_id))  # 将pad(id=0)的emb变为[0,0,...]
    return embedding, var


def proj_logits(outputs, hidden_size, logit_size, name='proj_logits', reuse=tf.AUTO_REUSE):
    """ if name = 'embedding' 复用embed矩阵 
        outputs [batch, length, hidden] or [batch, hidden]
    """

    with tf.variable_scope(name, reuse=reuse):
        var = tf.get_variable('weights', [logit_size, hidden_size],  # [vocab,hidden]
                              initializer=tf.random_normal_initializer(0.0, hidden_size ** -0.5))

    outputs_shape = shape_list(outputs)  # [batch, length, hidden]
    outputs = tf.reshape(outputs, [-1, outputs_shape[-1]])  # [batch*length,hidden]
    logits = tf.matmul(outputs, var, transpose_b=True)  # x,h * h,l -> x,l
    logits = tf.reshape(logits, outputs_shape[:-1] + [logit_size])  # [batch,length,vocab]

    return logits
