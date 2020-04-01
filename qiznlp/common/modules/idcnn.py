#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class IDCNN():
    def __init__(self, **kwargs):
        """
        """
        self.kernel_size = 3  # kernel_size
        self.num_filters = 100  # out_channel
        self.repeat_times = 4  # 共4个block
        self.layers = [  # 每个block结构
            {
                'dilation': 1
            },
            {
                'dilation': 3
            },
            {
                'dilation': 5
            },
        ]

    def __call__(self, embedding, name='idcnn', reuse=tf.AUTO_REUSE, **kwargs):
        """
        :param idcnn_inputs embedding: [batch, len, embed] 
        :return: [batch, len, num_filter * repeat_times]
        """
        with tf.variable_scope(name, reuse=reuse):
            inputs = tf.expand_dims(embedding, 1)  # [batch, 1, length, embed] 

            # shape of input = [batch, in_height, in_width, in_channels]
            layerinput = tf.layers.conv2d(inputs, self.num_filters, [1, self.kernel_size],
                                          strides=[1, 1], padding='SAME',
                                          use_bias=False,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='init_layer')

            final_out_from_layers = []
            total_width_for_last_dim = 0
            for j in range(self.repeat_times):  # 多个block共享参数
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=tf.AUTO_REUSE):
                        w = tf.get_variable("filterW", shape=[1, self.kernel_size, self.num_filters, self.num_filters],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filters])
                        layeroutput = tf.nn.atrous_conv2d(layerinput,
                                                          w,
                                                          rate=dilation,
                                                          padding="SAME")

                    layeroutput = tf.nn.bias_add(layeroutput, b)
                    layeroutput = tf.nn.relu(layeroutput)
                    if i == (len(self.layers) - 1):
                        final_out_from_layers.append(layeroutput)
                        total_width_for_last_dim += self.num_filters
                    layerinput = layeroutput
            idcnn_output = tf.concat(axis=3, values=final_out_from_layers)  # 【batch,1,len,hid]】
            idcnn_output = tf.squeeze(idcnn_output, 1)

            return idcnn_output  # [batch,len,hid]
