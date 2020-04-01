#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class Bi_RNN():
    def __init__(self, **kwargs):
        """
        """
        self.cell_name = kwargs['cell_name']  # 'GRUCell'/'LSTMCell'
        self.dropout = kwargs['dropout_rate']  # default 0. (0,1)
        self.hidden_size = kwargs['hidden_size']

        Cell = getattr(tf.nn.rnn_cell, self.cell_name)  # class GRUCell/LSTMCell
        self.fw_cell = Cell(self.hidden_size, name='fw')
        self.bw_cell = Cell(self.hidden_size, name='bw')

        if isinstance(self.dropout, tf.Tensor):  # 是placeholder则一定要先wrapper了
            self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)
            self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)
        else:
            if self.dropout:  # float not 0.
                self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)
                self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)

    def __call__(self, embedding, seq_length, name=None, reuse=tf.AUTO_REUSE, **kwargs):
        """
        :param inputs embedding: [batch, length, embed] 
        seq_length: [batch]
        :return: outputs: [batch, length, 2*hidden], state: [batch, 2*hidden]
        """
        scope_name = name + '/' + f'bi_{self.cell_name}_encoder' if name else f'bi_{self.cell_name}_encoder'
        with tf.variable_scope(scope_name, reuse=reuse):
            (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell,
                                                                                             embedding,
                                                                                             sequence_length=seq_length,
                                                                                             dtype=tf.float32)
            if self.cell_name == 'LSTMCell':
                # LSTMStateTuple
                fw_state = fw_state.h
                bw_state = bw_state.h
            
            outputs = tf.concat([fw_outputs, bw_outputs], axis=-1)  # [batch,length,2*hidden]
            state = tf.concat([fw_state, bw_state], axis=-1)  # [batch,2*hidden]

            return outputs, state


class RNN():
    def __init__(self, **kwargs):
        """
        """
        self.cell_name = kwargs['cell_name']  # 'GRUCell'/'LSTMCell'
        self.dropout = kwargs['dropout_rate']  # default 0. (0,1)
        self.hidden_size = kwargs['hidden_size']
        self.name = kwargs['name']

        Cell = getattr(tf.nn.rnn_cell, self.cell_name)  # class GRUCell/LSTMCell
        self.cell = Cell(self.hidden_size, name='rnn_cell')

        if isinstance(self.dropout, tf.Tensor):  # 是placeholder则一定要先wrapper了
            self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)
        else:
            if self.dropout:  # float not 0.
                self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, input_keep_prob=1. - self.dropout, output_keep_prob=1.0)

    def __call__(self, embedding, seq_length, reuse=tf.AUTO_REUSE, one_step=False, **kwargs):
        """
        :param inputs embedding: [batch, length, embed] 
        :return: outputs: [batch, length, 2*hidden], state: [batch, 2*hidden]
        """
        with tf.variable_scope(f'{self.name}_{self.cell_name}', reuse=reuse):
            outputs, state = tf.nn.dynamic_rnn(self.cell,
                                               embedding,
                                               sequence_length=seq_length,
                                               dtype=tf.float32)
            return outputs, state  # [batch,length,hidden] [batch,hidden]

    def one_step(self, one_step_input, state):
        with tf.variable_scope(f'{self.name}_{self.cell_name}/rnn', reuse=tf.AUTO_REUSE):
            output, state = self.cell(one_step_input, state)
        return output, state  # [batch,hidden] [batch,hidden]
