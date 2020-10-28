import os
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

from qiznlp.common.modules.common_layers import shape_list, mask_nonpad_from_embedding
from qiznlp.common.modules.embedding import embedding
from qiznlp.common.modules.encoder import conv_multi_kernel, batch_coattention_nnsubmulti
from qiznlp.common.modules.encoder import multihead_attention_encoder as multihead_attention
import qiznlp.common.utils as utils

conf = utils.dict2obj({
    'max_turn': 4,
    'max_char': 4,
    'vocab_size': 13124,
    'embed_size': 300,
    'char_vocab_size': 2943,
    'char_embed_size': 300,
    'hidden_size': 300,
    'num_heads': 6,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'pretrain_emb': None,
})


class Model(object):
    def __init__(self, build_graph=True, **kwargs):
        self.conf = conf
        self.run_model = kwargs.get('run_model', None)  # acquire outside run_model instance
        if build_graph:
            # build placeholder
            self.build_placeholder()
            # build model
            self.model_name = kwargs.get('model_name', 'MRFN_1')
            {
                # Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network https://github.com/julianser/hed-dlg
                'DAM': self.build_model1,  # https://www.aclweb.org/anthology/P18-1103/
                # Multi-Representation Fusion Network for Multi-Turn Response Selection in Retrieval-Based Chatbots https://github.com/chongyangtao/MRFN
                'MRFN': self.build_model2,  # https://dl.acm.org/doi/10.1145/3289600.3290985
                # personal modify: use word_char fuse as base embedding, but seem not convergent
                'MRFN_1': self.build_model3,
                # add new here
            }[self.model_name]()
            print(f'model_name: {self.model_name} build graph ok!')

    def build_placeholder(self):
        # placeholder
        # 原则上模型输入输出不变，不需换新model
        self.multi_s1 = tf.placeholder(tf.int32, [None, None, None], name='multi_s1')  # [batch,turn,len]
        self.s2 = tf.placeholder(tf.int32, [None, None], name='s2')  # [batch,len]

        # mrfn use char
        self.char_multi_s1 = tf.placeholder(tf.int32, [None, None, None, None], name='char_multi_s1')  # [batch,turn,len,char]
        self.char_s2 = tf.placeholder(tf.int32, [None, None, None], name='char_s2')  # [batch,len,char]

        self.target = tf.placeholder(tf.int32, [None], name='target')
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

    def build_model1(self):
        from qiznlp.common.modules.DAM import layers as layers
        from qiznlp.common.modules.DAM import operations as op

        fix_turn = conf.max_turn  # 4  # dam需显式指定s1的turn 因为使用了tf.unstack
        fix_len = 50  # dam需显式指定s1和s2的len 因为与最后立方体卷积的参数相关

        # 方式1:通过重写原来的placeholder, 此时要求输入数据符合shape, 不够灵活
        # self.multi_s1 = tf.placeholder(tf.int32, [None, 4, 50], name='multi_s1')  # [batch,turn,len]
        # self.s2 = tf.placeholder(tf.int32, [None, 50], name='s2')  # [batch,len]
        # multi_s1 = self.multi_s1
        # s2 = self.s2

        # 方式2:通过在原来的placeholder补齐来实现
        batch_size, s1_cur_turn, s1_cur_len = shape_list(self.multi_s1)
        multi_s1_pad_len = tf.zeros([batch_size, s1_cur_turn, fix_len - s1_cur_len], dtype=tf.int32)  # [batch,turn,-]
        multi_s1 = tf.concat([self.multi_s1, multi_s1_pad_len], -1)  # [batch,turn,50]
        multi_s1_pad_turn = tf.zeros([batch_size, fix_turn - s1_cur_turn, fix_len], dtype=tf.int32)  # [batch,-,50]
        multi_s1 = tf.concat([multi_s1, multi_s1_pad_turn], 1)  # [batch,4,50]
        multi_s1.set_shape([None, fix_turn, fix_len])

        s2_cur_len = shape_list(self.s2)[1]
        s2_pad = tf.zeros([batch_size, fix_len - s2_cur_len], dtype=tf.int32)  # [batch,-]
        s2 = tf.concat([self.s2, s2_pad], -1)  # [batch,50]
        s2.set_shape([None, fix_len])

        embed_size = conf.embed_size
        vocab_size = conf.vocab_size
        stack_num = 5

        is_positional = True

        batch_size, s1_turn, s1_length = shape_list(multi_s1)
        # ====== multi_s1 embedding
        # multi_s1_embed: [batch,turn,len,emb]
        multi_s1_embed, _ = embedding(tf.expand_dims(multi_s1, -1), vocab_size, embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        multi_s1_mask = mask_nonpad_from_embedding(multi_s1_embed)  # [batch,turn,len] 1 for nonpad; 0 for pad
        multi_s1_seqlen = tf.cast(tf.reduce_sum(multi_s1_mask, -1), tf.int32)  # [batch,turn]

        # ====== s2 embedding
        # s2_embed: [batch,len,emb] 
        s2_embed, _ = embedding(tf.expand_dims(s2, -1), vocab_size, embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]

        # s2
        Hr = s2_embed
        if is_positional and stack_num > 0:
            with tf.variable_scope('positional'):
                Hr = op.positional_encoding_vector(Hr, max_timescale=10)
        Hr = tf.layers.dropout(Hr, rate=self.dropout_rate)
        Hr_stack = [Hr]

        for index in range(stack_num):
            with tf.variable_scope('self_stack_%d' % index):
                Hr = layers.block(
                    Hr, Hr, Hr,
                    Q_lengths=s2_seqlen,
                    K_lengths=s2_seqlen,
                    drop_prob=self.dropout_rate)  # attention dropout
                Hr_stack.append(Hr)

        # multi_s1
        s1_embed_lst = tf.unstack(multi_s1_embed, num=s1_turn, axis=1)  # turn * [batch,len,emb]
        s1_seqlen_lst = tf.unstack(multi_s1_seqlen, num=s1_turn, axis=1)  # turn * [batch]

        sim_turns = []
        # for every turn calculate matching vector
        for s1_embed, s1_seqlen in zip(s1_embed_lst, s1_seqlen_lst):
            Hu = s1_embed  # [batch,len,emb]

            if is_positional and stack_num > 0:
                with tf.variable_scope('positional', reuse=True):
                    Hu = op.positional_encoding_vector(Hu, max_timescale=10)
            Hu = tf.layers.dropout(Hu, rate=self.dropout_rate)
            Hu_stack = [Hu]

            for index in range(stack_num):
                with tf.variable_scope('self_stack_%d' % index, reuse=True):
                    Hu = layers.block(
                        Hu, Hu, Hu,
                        Q_lengths=s1_seqlen,
                        K_lengths=s1_seqlen,
                        drop_prob=self.dropout_rate)  # attention dropout

                    Hu_stack.append(Hu)

            r_a_t_stack = []
            t_a_r_stack = []
            for index in range(stack_num + 1):
                with tf.variable_scope('t_attend_r_%d' % index, reuse=tf.AUTO_REUSE):
                    t_a_r = layers.block(
                        Hu_stack[index], Hr_stack[index], Hr_stack[index],
                        Q_lengths=s1_seqlen,
                        K_lengths=s2_seqlen,
                        drop_prob=self.dropout_rate)  # attention dropout

                with tf.variable_scope('r_attend_t_%d' % index, reuse=tf.AUTO_REUSE):
                    r_a_t = layers.block(
                        Hr_stack[index], Hu_stack[index], Hu_stack[index],
                        Q_lengths=s2_seqlen,
                        K_lengths=s1_seqlen,
                        drop_prob=self.dropout_rate)  # attention dropout

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            t_a_r = tf.stack(t_a_r_stack, axis=-1)
            r_a_t = tf.stack(r_a_t_stack, axis=-1)

            # calculate similarity matrix
            with tf.variable_scope('similarity'):
                # sim shape [batch, turn, len, 2*stack_num+1]
                # divide sqrt(200) to prevent gradient explosion
                sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(200.0)

            sim_turns.append(sim)

        # cnn and aggregation
        sim = tf.stack(sim_turns, axis=1)
        print('sim shape: %s' % sim.shape)
        with tf.variable_scope('cnn_aggregation'):
            final_info = layers.CNN_3d(sim, 32, 16)
            # for douban
            # final_info = layers.CNN_3d(sim, 16, 16)

        self.loss, self.logits = layers.loss(final_info, self.target)

        y_prob = tf.nn.sigmoid(self.logits)
        self.y_prob = tf.identity(y_prob, 'y_prob')

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(
                tf.cast(tf.greater_equal(self.y_prob, 0.5), tf.int32),
                self.target)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_model2(self):

        def interaction_matching_batch(s1_hidden, s2_hidden, hidden_size, s1_mask, s1_turn, scope="interaction_matching", reuse=None):
            # s1_hidden [batch*turn,len,emb]
            # s2_hidden [batch*turn,len,emb]
            with tf.variable_scope(scope, reuse=reuse):
                res_coatt = batch_coattention_nnsubmulti(s1_hidden,
                                                         s2_hidden,
                                                         tf.to_float(s1_mask),
                                                         scope='%s_res' % scope)

                res_coatt = tf.layers.dropout(res_coatt, self.dropout_rate)  # [batch*turn,len,hid]

                # 第一级rnn 规约句子
                res_coatt_cell = tf.contrib.rnn.GRUCell(hidden_size)
                _, res_final = tf.nn.dynamic_rnn(res_coatt_cell, res_coatt, dtype=tf.float32, scope='first_level_gru')

                res_feature = res_final  # [batch*turn,hid]
                res_feature = tf.reshape(res_feature, [-1, s1_turn, hidden_size])  # [batch,turn,hid]

                # 第二级rnn 规约多轮
                final_gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
                _, last_hidden = tf.nn.dynamic_rnn(final_gru_cell, res_feature, dtype=tf.float32, scope='final_gru')
                logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
                return logits

        self.loss_lst = []
        self.y_prob_lst = []

        use_char = True
        use_word = True
        use_seq = True
        use_conv = True
        use_self = True
        use_cross = True

        embed_size = conf.embed_size
        vocab_size = conf.vocab_size
        cembed_size = conf.char_embed_size
        cvocab_size = conf.char_vocab_size

        batch_size, s1_turn, s1_length = shape_list(self.multi_s1)
        char_len = shape_list(self.char_multi_s1)[3]  # 4 here

        # ====== multi_s1 embedding
        # _norm mean reshape to batch * turn
        # multi_s1_embed: [batch,turn,len,emb]
        multi_s1_embed, _ = embedding(tf.expand_dims(self.multi_s1, -1), vocab_size, embed_size, name='share_word_embedding', pretrain_embedding=conf.pretrain_emb)
        multi_s1_mask = mask_nonpad_from_embedding(multi_s1_embed)  # [batch,turn,len] 1 for nonpad; 0 for pad
        multi_s1_mask_norm = tf.reshape(multi_s1_mask, [-1, s1_length])  # [batch*turn,len]
        multi_s1_seqlen = tf.cast(tf.reduce_sum(multi_s1_mask, -1), tf.int32)  # [batch,turn]
        multi_s1_seqlen_norm = tf.reshape(multi_s1_seqlen, [-1])  # [batch*turn]
        multi_s1_embed_norm = tf.reshape(multi_s1_embed, [-1, s1_length, embed_size])  # [batch*turn,len,emb]

        # ====== s2 embedding
        s2_length = shape_list(self.s2)[1]
        # _turn mean s2 expand and tile to batch,turn,..
        # _norm mean s2 reshape to batch*turn
        # s2_embed: [batch,len,emb] 
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), vocab_size, embed_size, name='share_word_embedding', pretrain_embedding=conf.pretrain_emb)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_mask_turn = tf.tile(tf.expand_dims(s2_mask, 1), [1, s1_turn, 1])  # [batch,turn,len]
        s2_mask_norm = tf.reshape(s2_mask_turn, [-1, s2_length])  # [batch*turn,len]
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]
        s2_embed_turn = tf.tile(tf.expand_dims(s2_embed, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,emb]
        s2_embed_norm = tf.reshape(s2_embed_turn, [-1, s2_length, embed_size])  # [batch*turn,len,emb]

        # ====== representation and interaction

        # ====== char representation
        if use_char:
            char_filter = 100
            kernels_size = [2, 3, 4]

            # ====== multi_s1 char embedding
            # _norm mean reshape to batch*turn
            # s1_char_embed: [batch,turn,len,char,emb]
            char_multi_s1_embed, _ = embedding(tf.expand_dims(self.char_multi_s1, -1), cvocab_size, cembed_size, name='share_char_embedding')
            char_multi_s1_mask = mask_nonpad_from_embedding(char_multi_s1_embed)  # [batch,turn,len,4] 1 for nonpad; 0 for pad
            char_multi_s1_embed = tf.reshape(char_multi_s1_embed, [-1, char_len, cembed_size])  # [batch*turn*len,4,emb]
            char_multi_s1_conv = conv_multi_kernel(char_multi_s1_embed, char_filter, kernels_size=kernels_size,
                                                   padding='same', use_bias=True, activation=tf.nn.relu,
                                                   name="char_conv", layer_norm_finally=False)  # [batch*turn*len,4,filter*3]
            # remove_pad 将pad的vector变为极小
            char_multi_s1_conv += tf.expand_dims(tf.reshape(1. - char_multi_s1_mask, [-1, char_len]), -1) * -1e9  # [batch*turn*len,4,1]
            # max-pooling
            char_multi_s1_conv = tf.reduce_max(char_multi_s1_conv, axis=1)  # [batch*turn*len,hid]
            char_multi_s1_conv = tf.reshape(char_multi_s1_conv, [-1, s1_turn, s1_length, char_filter * len(kernels_size)])  # [batch,turn,len,hid]
            char_multi_s1_embed = tf.layers.dropout(char_multi_s1_conv, self.dropout_rate)
            char_multi_s1_embed_norm = tf.reshape(char_multi_s1_embed, [-1, s1_length, char_filter * len(kernels_size)])  # [batch*turn,len,hid]

            # ====== s2 char embedding
            # _turn mean expand and tile to batch,turn,..
            # _norm mean reshape to batch*turn
            # char_s2_embed: [batch,len,4,emb]
            char_s2_embed, _ = embedding(tf.expand_dims(self.char_s2, -1), cvocab_size, cembed_size, name='share_char_embedding')
            char_s2_mask = mask_nonpad_from_embedding(char_s2_embed)  # [batch,len,4] 1 for nonpad; 0 for pad
            char_s2_embed = tf.reshape(char_s2_embed, [-1, char_len, cembed_size])  # [batch*len,4,emb]
            char_s2_conv = conv_multi_kernel(char_s2_embed, char_filter, kernels_size=kernels_size,
                                             padding='same', use_bias=True, activation=tf.nn.relu,
                                             name="char_conv", layer_norm_finally=False)  # [batch*len,4,filter*3]
            # remove_pad 将pad的vector变为极小
            char_s2_conv += tf.expand_dims(tf.reshape(1. - char_s2_mask, [-1, char_len]), -1) * -1e9  # [batch*len,4,1]
            # max-pooling
            char_s2_conv = tf.reduce_max(char_s2_conv, axis=1)  # [batch*len,hid]
            char_s2_conv = tf.reshape(char_s2_conv, [-1, s2_length, char_filter * len(kernels_size)])  # [batch,len,hid]
            char_s2_embed = tf.layers.dropout(char_s2_conv, self.dropout_rate)
            char_s2_embed_turn = tf.tile(tf.expand_dims(char_s2_embed, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,hid]
            char_s2_embed_norm = tf.reshape(char_s2_embed_turn, [-1, s2_length, char_filter * len(kernels_size)])  # [batch*turn,len,hid]

            char_interaction = interaction_matching_batch(char_multi_s1_embed_norm,
                                                          char_s2_embed_norm,
                                                          embed_size,
                                                          multi_s1_mask_norm,
                                                          s1_turn,
                                                          scope='char_interaction_matching')
            loss_char = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=char_interaction))
            self.loss_lst.append(loss_char)
            self.y_prob_lst.append(tf.nn.softmax(char_interaction))

        # ====== word representation
        if use_word:
            word_interaction = interaction_matching_batch(multi_s1_embed_norm,
                                                          s2_embed_norm,
                                                          embed_size,
                                                          multi_s1_mask_norm,
                                                          s1_turn,
                                                          scope='word_interaction_matching')
            loss_word = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=word_interaction))
            self.loss_lst.append(loss_word)
            self.y_prob_lst.append(tf.nn.softmax(word_interaction))

        # ====== rnn representation
        rnn_dim = 300
        if use_seq:
            with tf.variable_scope("rnn_embeddings"):
                sentence_gru_cell = tf.contrib.rnn.GRUCell(rnn_dim)
                with tf.variable_scope('sentence_gru'):
                    # [batch*turn,len,hid]
                    s1_gru_hidden_norm, _ = tf.nn.dynamic_rnn(sentence_gru_cell, multi_s1_embed_norm, sequence_length=multi_s1_seqlen_norm, dtype=tf.float32)

                with tf.variable_scope('sentence_gru', reuse=True):
                    # [batch,len,hid]
                    s2_gru_hidden, _ = tf.nn.dynamic_rnn(sentence_gru_cell, s2_embed, sequence_length=s2_seqlen, dtype=tf.float32)

            s2_gru_hidden_turn = tf.tile(tf.expand_dims(s2_gru_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,hid]
            s2_gru_hidden_norm = tf.reshape(s2_gru_hidden_turn, [-1, s2_length, rnn_dim])  # [batch*turn,len,hid]

            seq_interaction = interaction_matching_batch(s1_gru_hidden_norm,
                                                         s2_gru_hidden_norm,
                                                         rnn_dim,
                                                         multi_s1_mask_norm,
                                                         s1_turn,
                                                         scope='seg_interaction_matching')
            loss_seq = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=seq_interaction))
            self.loss_lst.append(loss_seq)
            self.y_prob_lst.append(tf.nn.softmax(seq_interaction))

        # ====== conv representation
        if use_conv:
            conv_dim = 100
            kernels = [1, 2, 3, 4]
            with tf.variable_scope("conv_embeddings"):
                s1_conv_hidden_norm = conv_multi_kernel(multi_s1_embed_norm, conv_dim, kernels_size=kernels,
                                                        padding='SAME', use_bias=True, activation=tf.nn.relu,
                                                        name="seq_conv", layer_norm_finally=True)  # [batch*turn,len,filter*4]

                s2_conv_hidden = conv_multi_kernel(s2_embed, conv_dim, kernels_size=kernels,
                                                   padding='SAME', use_bias=True, activation=tf.nn.relu,
                                                   name="seq_conv", layer_norm_finally=True)  # [batch,len,filter*4]

                s2_conv_hidden_turn = tf.tile(tf.expand_dims(s2_conv_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,filter*4]
                s2_conv_hidden_norm = tf.reshape(s2_conv_hidden_turn, [-1, s2_length, conv_dim * len(kernels)])  # [batch*turn,len,filter*4]

            conv_interaction = interaction_matching_batch(s1_conv_hidden_norm,
                                                          s2_conv_hidden_norm,
                                                          conv_dim * len(kernels),
                                                          multi_s1_mask_norm,
                                                          s1_turn,
                                                          scope='conv_interaction_matching')
            loss_conv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=conv_interaction))
            self.loss_lst.append(loss_conv)
            self.y_prob_lst.append(tf.nn.softmax(conv_interaction))

        # ====== self attention
        if use_self:
            with tf.variable_scope("self_att_embeddings"):
                s1_self_att_hidden_norm = multihead_attention(multi_s1_embed_norm,
                                                              multi_s1_embed_norm,
                                                              multi_s1_mask_norm,
                                                              embed_size,
                                                              conf.num_heads,
                                                              self.dropout_rate)

                s2_self_att_hidden = multihead_attention(s2_embed,
                                                         s2_embed,
                                                         s2_mask,
                                                         embed_size,
                                                         conf.num_heads,
                                                         self.dropout_rate)

                s2_self_att_hidden_turn = tf.tile(tf.expand_dims(s2_self_att_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,hid]
                s2_self_att_hidden_norm = tf.reshape(s2_self_att_hidden_turn, [-1, s2_length, embed_size])  # [batch*turn,len,hid]

            self_att_interaction = interaction_matching_batch(s1_self_att_hidden_norm,
                                                              s2_self_att_hidden_norm,
                                                              embed_size,
                                                              multi_s1_mask_norm,
                                                              s1_turn,
                                                              scope='self_att_interaction_matching')
            loss_self = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self_att_interaction))
            self.loss_lst.append(loss_self)
            self.y_prob_lst.append(tf.nn.softmax(self_att_interaction))

        # cross attention
        if use_cross:
            with tf.variable_scope("cross_att_embeddings"):
                s1_cross_att_hidden_norm = multihead_attention(multi_s1_embed_norm,
                                                               s2_embed_norm,
                                                               s2_mask_norm,
                                                               embed_size,
                                                               conf.num_heads,
                                                               self.dropout_rate)

                s2_cross_attn_hidden_norm = multihead_attention(s2_embed_norm,
                                                                multi_s1_embed_norm,
                                                                multi_s1_mask_norm,
                                                                embed_size,
                                                                conf.num_heads,
                                                                self.dropout_rate)

                # s2_cross_attn_hidden_lst = []
                # self.response_cross_att_embeddings = []
                # s1_embed_lst = tf.unstack(multi_s1_embed, axis=1)  # turn * [batch,len,emb]
                # s1_mask_lst = tf.unstack(multi_s1_mask, axis=1)  # turn * [batch,len]
                # for k, s1_embed_ in enumerate(s1_embed_lst):
                #     s2_cross_att_hidden = multihead_attention(s2_embed,
                #                                               s1_embed_,
                #                                               s1_mask_lst[k],
                #                                               embed_size,
                #                                               conf.num_heads,
                #                                               self.dropout_rate)
                #     s2_cross_attn_hidden_lst.append(s2_cross_att_hidden)  # turn * [batch,len,hid]
                # 
                # s2_cross_attn_hidden_turn = tf.stack(s2_cross_attn_hidden_lst, axis=1)  # [batch,turn,len,hid]
                # s2_cross_attn_hidden_norm = tf.reshape(s2_cross_attn_hidden_turn, [-1, s2_length, embed_size])

            cross_att_interaction = interaction_matching_batch(s1_cross_att_hidden_norm,
                                                               s2_cross_attn_hidden_norm,
                                                               embed_size,
                                                               multi_s1_mask_norm,
                                                               s1_turn,
                                                               scope='cross_att_interaction_matching')
            loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=cross_att_interaction))
            self.loss_lst.append(loss_cross)
            self.y_prob_lst.append(tf.nn.softmax(cross_att_interaction))

        self.loss = sum(self.loss_lst)
        y_prob = sum(self.y_prob_lst)  # [batch,2]
        y_prob = y_prob[:, 1]
        self.y_prob = tf.identity(y_prob, 'y_prob')

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(
                tf.cast(tf.greater_equal(self.y_prob, 3), tf.int32),  # 因为加了6个0-1的概率，随意中值用2.5
                self.target)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_model3(self):

        def interaction_matching_batch(s1_hidden, s2_hidden, hidden_size, s1_mask, s1_turn, scope="interaction_matching", reuse=None):
            # s1_hidden [batch*turn,len,emb]
            # s2_hidden [batch*turn,len,emb]
            with tf.variable_scope(scope, reuse=reuse):
                res_coatt = batch_coattention_nnsubmulti(s1_hidden,
                                                         s2_hidden,
                                                         tf.to_float(s1_mask),
                                                         scope='%s_res' % scope)

                res_coatt = tf.layers.dropout(res_coatt, self.dropout_rate)  # [batch*turn,len,hid]

                # 第一级rnn 规约句子
                res_coatt_cell = tf.contrib.rnn.GRUCell(hidden_size)
                _, res_final = tf.nn.dynamic_rnn(res_coatt_cell, res_coatt, dtype=tf.float32, scope='first_level_gru')

                res_feature = res_final  # [batch*turn,hid]
                res_feature = tf.reshape(res_feature, [-1, s1_turn, hidden_size])  # [batch,turn,hid]

                # 第二级rnn 规约多轮
                final_gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
                _, last_hidden = tf.nn.dynamic_rnn(final_gru_cell, res_feature, dtype=tf.float32, scope='final_gru')
                logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
                return logits

        self.loss_lst = []
        self.y_prob_lst = []

        use_word = True
        use_seq = True
        use_conv = True
        use_self = True
        use_cross = True

        embed_size = conf.embed_size
        vocab_size = conf.vocab_size
        cembed_size = conf.char_embed_size
        cvocab_size = conf.char_vocab_size

        batch_size, s1_turn, s1_length = shape_list(self.multi_s1)
        char_len = shape_list(self.char_multi_s1)[3]  # 4 here

        # ====== multi_s1 embedding
        # norm mean reshape to batch * turn
        # multi_s1_embed: [batch,turn,len,emb]
        multi_s1_embed, _ = embedding(tf.expand_dims(self.multi_s1, -1), vocab_size, embed_size, name='share_word_embedding', pretrain_embedding=conf.pretrain_emb)
        multi_s1_mask = mask_nonpad_from_embedding(multi_s1_embed)  # [batch,turn,len] 1 for nonpad; 0 for pad
        multi_s1_mask_norm = tf.reshape(multi_s1_mask, [-1, s1_length])  # [batch*turn,len]
        multi_s1_seqlen = tf.cast(tf.reduce_sum(multi_s1_mask, -1), tf.int32)  # [batch,turn]
        multi_s1_seqlen_norm = tf.reshape(multi_s1_seqlen, [-1])  # [batch*turn]

        # ====== multi_s1 char embedding
        # s1_char_embed: [batch,turn,len,char,emb]
        char_multi_s1_embed, _ = embedding(tf.expand_dims(self.char_multi_s1, -1), cvocab_size, cembed_size, name='share_char_embedding')
        char_multi_s1_mask = mask_nonpad_from_embedding(char_multi_s1_embed)  # [batch,turn,len,4] 1 for nonpad; 0 for pad
        char_multi_s1_embed = tf.reshape(char_multi_s1_embed, [-1, char_len, cembed_size])  # [batch*turn*len,4,emb]
        
        # 方式1:char cnn + max-pooling
        # char_filter = 100
        # kernels_size = [2, 3, 4]
        # char_multi_s1_conv = conv_multi_kernel(char_multi_s1_embed, char_filter, kernels_size=kernels_size,
        #                                        padding='same', use_bias=True, activation=tf.nn.relu,
        #                                        name="char_conv", layer_norm_finally=False)  # [batch*turn*len,4,filter*3]
        # # max-pooling
        # # remove_pad 将pad的vector变为极小
        # char_multi_s1_conv += tf.expand_dims(tf.reshape(1. - char_multi_s1_mask, [-1, char_len]), -1) * -1e9  # [batch*turn*len,4,1]
        # char_multi_s1_conv = tf.reduce_max(char_multi_s1_conv, axis=1)  # [batch*turn*len,hid]
        # char_multi_s1_conv = tf.reshape(char_multi_s1_conv, [-1, s1_turn, s1_length, char_filter * len(kernels_size)])  # [batch,turn,len,hid]
        # char_multi_s1_conv = tf.layers.dropout(char_multi_s1_conv, self.dropout_rate)
        
        # 方式2:char mean-pooling
        char_multi_s1_sum = tf.reduce_sum(char_multi_s1_embed, axis=-2)  # [batch*turn*len,hid]
        char_multi_s1_num = tf.reduce_sum(tf.reshape(char_multi_s1_mask, [-1, char_len]), -1, keepdims=True)  # [batch*turn*len,1]
        char_multi_s1_mean = char_multi_s1_sum / char_multi_s1_num  #   # [batch*turn*len,hid]
        
        char_multi_s1_mean = tf.reshape(char_multi_s1_mean, [-1, s1_turn, s1_length, cembed_size])  # [batch,turn,len,hid]
        # char_multi_s1_mean = tf.layers.dropout(char_multi_s1_mean, self.dropout_rate)

        # ====== fuse word and char embedding for multi_s1
        multi_s1_embed += char_multi_s1_mean  # seem not convergence -_-
        # multi_s1_embed = tf.concat([multi_s1_embed, char_multi_s1_mean], axis=-1)  # [batch,turn,len,emb+]
        # [batch,turn,len,300]
        # multi_s1_embed = tf.layers.dense(multi_s1_embed, 300, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='word_fuse', reuse=tf.AUTO_REUSE)
        multi_s1_embed_norm = tf.reshape(multi_s1_embed, [-1, s1_length, 300])  # [batch*turn,len,300]

        # ====== s2 embedding
        s2_length = shape_list(self.s2)[1]
        # _turn mean s2 expand and tile to batch,turn,..
        # _norm mean s2 reshape to batch*turn
        # s2_embed: [batch,len,emb] 
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), vocab_size, embed_size, name='share_word_embedding', pretrain_embedding=conf.pretrain_emb)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_mask_turn = tf.tile(tf.expand_dims(s2_mask, 1), [1, s1_turn, 1])  # [batch,turn,len]
        s2_mask_norm = tf.reshape(s2_mask_turn, [-1, s2_length])  # [batch*turn,len]
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]

        # ====== s2 char embedding
        # char_s2_embed: [batch,len,4,emb]
        char_s2_embed, _ = embedding(tf.expand_dims(self.char_s2, -1), cvocab_size, cembed_size, name='share_char_embedding')
        char_s2_mask = mask_nonpad_from_embedding(char_s2_embed)  # [batch,len,4] 1 for nonpad; 0 for pad
        char_s2_embed = tf.reshape(char_s2_embed, [-1, char_len, cembed_size])  # [batch*len,4,emb]

        # 方式1:char cnn + max-pooling
        # char_s2_conv = conv_multi_kernel(char_s2_embed, char_filter, kernels_size=kernels_size,
        #                                  padding='same', use_bias=True, activation=tf.nn.relu,
        #                                  name="char_conv", layer_norm_finally=False)  # [batch*len,4,filter*3]
        # # max-pooling
        # # remove_pad 将pad的vector变为极小
        # char_s2_conv += tf.expand_dims(tf.reshape(1. - char_s2_mask, [-1, char_len]), -1) * -1e9  # [batch*len,4,1]
        # char_s2_conv = tf.reduce_max(char_s2_conv, axis=1)  # [batch*len,hid]
        # char_s2_conv = tf.reshape(char_s2_conv, [-1, s2_length, char_filter * len(kernels_size)])  # [batch,len,hid]
        # char_s2_conv = tf.layers.dropout(char_s2_conv, self.dropout_rate)

        # 方式2:char mean-pooling
        char_s2_sum = tf.reduce_sum(char_s2_embed, axis=-2)  # [batch*len,hid]
        char_s2_num = tf.reduce_sum(tf.reshape(char_s2_mask, [-1, char_len]), -1, keepdims=True)  # [batch*len,1]
        char_s2_mean = char_s2_sum / char_s2_num  # # [batch*len,hid]

        char_s2_mean = tf.reshape(char_s2_mean, [-1, s2_length, cembed_size])  # [batch,len,hid]
        # char_s2_mean = tf.layers.dropout(char_s2_mean, self.dropout_rate)

        # ====== fuse word and char embedding for s2
        s2_embed += char_s2_mean  # seem not convergence -_-
        # s2_embed = tf.concat([s2_embed, char_s2_mean], axis=-1)  # [batch,len,emb+]
        # [batch,len,300]
        # s2_embed = tf.layers.dense(s2_embed, 300, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='word_fuse', reuse=tf.AUTO_REUSE)
        s2_embed_turn = tf.tile(tf.expand_dims(s2_embed, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,300]
        s2_embed_norm = tf.reshape(s2_embed_turn, [-1, s2_length, 300])  # [batch*turn,len,300]

        # ====== representation and interaction

        # ====== word representation
        if use_word:
            word_interaction = interaction_matching_batch(multi_s1_embed_norm,
                                                          s2_embed_norm,
                                                          embed_size,
                                                          multi_s1_mask_norm,
                                                          s1_turn,
                                                          scope='word_interaction_matching')
            loss_word = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=word_interaction))
            self.loss_lst.append(loss_word)
            self.y_prob_lst.append(tf.nn.softmax(word_interaction))

        # ====== rnn representation
        rnn_dim = 300
        if use_seq:
            with tf.variable_scope("rnn_embeddings"):
                sentence_gru_cell = tf.contrib.rnn.GRUCell(rnn_dim)
                with tf.variable_scope('sentence_gru'):
                    # [batch*turn,len,hid]
                    s1_gru_hidden_norm, _ = tf.nn.dynamic_rnn(sentence_gru_cell, multi_s1_embed_norm, sequence_length=multi_s1_seqlen_norm, dtype=tf.float32)

                with tf.variable_scope('sentence_gru', reuse=True):
                    # [batch,len,hid]
                    s2_gru_hidden, _ = tf.nn.dynamic_rnn(sentence_gru_cell, s2_embed, sequence_length=s2_seqlen, dtype=tf.float32)

            s2_gru_hidden_turn = tf.tile(tf.expand_dims(s2_gru_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,hid]
            s2_gru_hidden_norm = tf.reshape(s2_gru_hidden_turn, [-1, s2_length, rnn_dim])  # [batch*turn,len,hid]

            seq_interaction = interaction_matching_batch(s1_gru_hidden_norm,
                                                         s2_gru_hidden_norm,
                                                         rnn_dim,
                                                         multi_s1_mask_norm,
                                                         s1_turn,
                                                         scope='seg_interaction_matching')
            loss_seq = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=seq_interaction))
            self.loss_lst.append(loss_seq)
            self.y_prob_lst.append(tf.nn.softmax(seq_interaction))

        # ====== conv representation
        if use_conv:
            conv_dim = 50
            kernels = [1, 2, 3, 4]
            with tf.variable_scope("conv_embeddings"):
                s1_conv_hidden_norm = conv_multi_kernel(multi_s1_embed_norm, conv_dim, kernels_size=kernels,
                                                        padding='same', use_bias=True, activation=tf.nn.relu,
                                                        name="seq_conv", layer_norm_finally=True)  # [batch*turn,len,filter*4]

                s2_conv_hidden = conv_multi_kernel(s2_embed, conv_dim, kernels_size=kernels,
                                                   padding='same', use_bias=True, activation=tf.nn.relu,
                                                   name="seq_conv", layer_norm_finally=True)  # [batch,len,filter*4]

                s2_conv_hidden_turn = tf.tile(tf.expand_dims(s2_conv_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,filter*4]
                s2_conv_hidden_norm = tf.reshape(s2_conv_hidden_turn, [-1, s2_length, conv_dim * len(kernels)])  # [batch*turn,len,filter*4]

            conv_interaction = interaction_matching_batch(s1_conv_hidden_norm,
                                                          s2_conv_hidden_norm,
                                                          conv_dim * len(kernels),
                                                          multi_s1_mask_norm,
                                                          s1_turn,
                                                          scope='conv_interaction_matching')
            loss_conv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=conv_interaction))
            self.loss_lst.append(loss_conv)
            self.y_prob_lst.append(tf.nn.softmax(conv_interaction))

        # ====== self attention
        if use_self:
            with tf.variable_scope("self_att_embeddings"):
                s1_self_att_hidden_norm = multihead_attention(multi_s1_embed_norm,
                                                              multi_s1_embed_norm,
                                                              multi_s1_mask_norm,
                                                              embed_size,
                                                              conf.num_heads,
                                                              self.dropout_rate)

                s2_self_att_hidden = multihead_attention(s2_embed,
                                                         s2_embed,
                                                         s2_mask,
                                                         embed_size,
                                                         conf.num_heads,
                                                         self.dropout_rate)

                s2_self_att_hidden_turn = tf.tile(tf.expand_dims(s2_self_att_hidden, 1), [1, s1_turn, 1, 1])  # [batch,turn,len,hid]
                s2_self_att_hidden_norm = tf.reshape(s2_self_att_hidden_turn, [-1, s2_length, embed_size])  # [batch*turn,len,hid]

            self_att_interaction = interaction_matching_batch(s1_self_att_hidden_norm,
                                                              s2_self_att_hidden_norm,
                                                              embed_size,
                                                              multi_s1_mask_norm,
                                                              s1_turn,
                                                              scope='self_att_interaction_matching')
            loss_self = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self_att_interaction))
            self.loss_lst.append(loss_self)
            self.y_prob_lst.append(tf.nn.softmax(self_att_interaction))

        # cross attention
        if use_cross:
            with tf.variable_scope("cross_att_embeddings"):
                s1_cross_att_hidden_norm = multihead_attention(multi_s1_embed_norm,
                                                               s2_embed_norm,
                                                               s2_mask_norm,
                                                               embed_size,
                                                               conf.num_heads,
                                                               self.dropout_rate)

                s2_cross_attn_hidden_norm = multihead_attention(s2_embed_norm,
                                                                multi_s1_embed_norm,
                                                                multi_s1_mask_norm,
                                                                embed_size,
                                                                conf.num_heads,
                                                                self.dropout_rate)

            cross_att_interaction = interaction_matching_batch(s1_cross_att_hidden_norm,
                                                               s2_cross_attn_hidden_norm,
                                                               embed_size,
                                                               multi_s1_mask_norm,
                                                               s1_turn,
                                                               scope='cross_att_interaction_matching')
            loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=cross_att_interaction))
            self.loss_lst.append(loss_cross)
            self.y_prob_lst.append(tf.nn.softmax(cross_att_interaction))

        self.loss = sum(self.loss_lst)
        y_prob = sum(self.y_prob_lst)  # [batch,2]
        y_prob = y_prob[:, 1]
        self.y_prob = tf.identity(y_prob, 'y_prob')

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(
                tf.cast(tf.greater_equal(self.y_prob, 2.5), tf.int32),  # 因为加了5个0-1的概率，随意中值用2.5
                self.target)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    @classmethod
    def sent2ids(cls, sent, word2id, max_word_len=None):
        # sent: 已分好词 ' '隔开
        # 形成batch时才动态补齐长度
        words = sent.split(' ')
        token_ids = [word2id.get(word, word2id['<unk>']) for word in words]
        if max_word_len:
            token_ids = token_ids[:max_word_len - 1]
        token_ids.append(word2id['<eos>'])
        # token_ids = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
        return token_ids  # [len]

    @classmethod
    def multi_sent2ids(cls, multi_sent, word2id, max_word_len=None, max_turn=None):
        # multi_sent: sent的list
        # sent: 已分好词 ' '隔开
        # 不能在形成batch时才动态补齐长度和轮数，因为tfrecord存储会平展为一维，并没有足够信息恢复为不定的[turn,len]
        # 至少需要固定一个(turn或len),后面才能恢复,且需要补齐为矩阵turn * len,后面才可reshape恢复
        # 这里固定turn
        multi_token_ids = []
        for sent in multi_sent:
            multi_token_ids.append(cls.sent2ids(sent, word2id, max_word_len=max_word_len))
        if max_turn is None:  # turn需固定来补齐句子长度
            max_turn = conf.max_turn
        multi_token_ids = multi_token_ids[-max_turn:]  # 若大于截取后n轮
        multi_token_ids = multi_token_ids + [[]] * (max_turn - len(multi_token_ids))  # 若不足则后面补齐
        multi_token_ids = utils.pad_sequences(multi_token_ids, padding='post', value=0)  # 补齐每个sent长度
        return multi_token_ids  # [turn(5), len]

    @classmethod
    def sent2ids_char(cls, sent, word2id, char2id, max_word_len=None, max_char=None, pad_eos=False):
        # sent: 已分好词 ' '隔开
        # char固定,形成batch时才动态补齐len
        words = sent.split(' ')

        if max_word_len:
            if pad_eos:
                words = words[:max_word_len - 1]  # 留末尾放<eos>
            else:
                words = words[:max_word_len]  # 留末尾放<eos>

        token_ids = [word2id.get(word, word2id['<unk>']) for word in words]  # [len]

        if max_char is None:
            max_char = conf.max_char
        char_token_ids = []  # [len,char]
        for word in words:
            char_ids = [char2id.get(char, char2id['<unk>']) for char in word[-max_char:]]  # 超长优先截取后面
            char_ids += [char2id['<pad>']] * (max_char - len(char_ids))  # 不足后面补齐0
            char_token_ids.append(char_ids)

        if pad_eos:
            # 末位补<eos>
            token_ids.append(word2id['<eos>'])
            char_token_ids.append([char2id['<pad>']] * max_char)  # <eos>的char用char_pad补齐

        assert len(token_ids) == len(char_token_ids)
        return token_ids, char_token_ids  # [len]  [len,char]

    @classmethod
    def multi_sent2ids_char(cls, multi_sent, word2id, char2id, max_word_len=None, max_turn=None, max_char=None):
        # 返回char
        # multi_sent: sent的list
        # sent: 已分好词 ' '隔开
        # 不能在形成batch时才动态补齐所有turn,len,char，因为tfrecord存储会平展为一维，并没有足够信息恢复为不定的[turn,len,char]
        # 至少需要固定turn和char,补齐为矩阵turn*len*char,后面才可reshape恢复
        # 这里固定turn/char 让len动态
        if max_char is None:  # 固定且补齐char
            max_char = conf.max_char
        if max_turn is None:  # 固定且补齐turn
            max_turn = conf.max_turn
        multi_token_ids = []
        multi_char_token_ids = []
        for sent in multi_sent:
            token_ids, char_token_ids = cls.sent2ids_char(sent, word2id, char2id, max_word_len=max_word_len, max_char=max_char)
            multi_token_ids.append(token_ids)  # [turn,len]
            multi_char_token_ids.append(char_token_ids)  # [turn,len,char]

        # 补齐turn
        multi_token_ids = multi_token_ids[-max_turn:]  # 若大于截取后n轮
        multi_token_ids = multi_token_ids + [[]] * (max_turn - len(multi_token_ids))  # 若不足则后面补齐
        multi_token_ids = utils.pad_sequences(multi_token_ids, padding='post', value=0)  # 补齐每个sent长度

        multi_char_token_ids = multi_char_token_ids[-max_turn:]  # 若大于截取后n轮
        multi_char_token_ids = multi_char_token_ids + [[]] * (max_turn - len(multi_char_token_ids))  # 若不足则后面补齐
        multi_char_token_ids = utils.pad_sequences(multi_char_token_ids, padding='post', value=0)  # 补齐每个sent长度

        return multi_token_ids, multi_char_token_ids  # [turn(5),len] [turn,len,char(4)]

    def create_feed_dict_from_data(self, data, ids, mode='train'):
        # data:数据已经转为id, data不同字段保存该段字段全量数据
        batch_multi_s1 = [data['multi_s1'][i] for i in ids]
        batch_s2 = [data['s2'][i] for i in ids]
        batch_target = [data['target'][i] for i in ids]
        batch_char_multi_s1 = [data['char_multi_s1'][i] for i in ids]
        batch_char_s2 = [data['char_s2'][i] for i in ids]

        # 多轮必定需要补齐，就不做判断了
        batch_multi_s1 = utils.pad_sequences(batch_multi_s1, padding='post')
        batch_char_multi_s1 = utils.pad_sequences(batch_char_multi_s1, padding='post')
        batch_char_s2 = utils.pad_sequences(batch_char_s2, padding='post')

        if len(set([len(e) for e in batch_s2])) != 1:  # 长度不等
            batch_s2 = utils.pad_sequences(batch_s2, padding='post')
        feed_dict = {
            self.multi_s1: batch_multi_s1,
            self.s2: batch_s2,
            self.target: batch_target,
            self.char_multi_s1: batch_char_multi_s1,
            self.char_s2: batch_char_s2,
        }
        if mode == 'train': feed_dict['num'] = len(batch_s2)
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_features(self, features, mode='train'):
        # feature:tfrecord数据的example, 每个features的不同字段包括该字段一个batch数据
        feed_dict = {
            self.multi_s1: features['multi_s1'],
            self.s2: features['s2'],
            self.target: features['target'],
            self.char_multi_s1: features['char_multi_s1'],
            self.char_s2: features['char_s2'],
        }
        if mode == 'train': feed_dict['num'] = len(features['s2'])
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_raw(self, batch_multi_s1, batch_s2, batch_target, token2id_dct, mode='infer'):
        word2id = token2id_dct['word2id']
        char2id = token2id_dct['char2id']

        feed_multi_s1 = []
        feed_char_multi_s1 = []
        for multi_s1 in batch_multi_s1:
            sent_lst = multi_s1.split('$$$')
            multi_token_ids, multi_char_token_ids = self.multi_sent2ids_char(sent_lst, word2id, char2id)
            feed_multi_s1.append(multi_token_ids)
            feed_char_multi_s1.append(multi_char_token_ids)

        feed_s2 = []
        feed_char_s2 = []
        for s2 in batch_s2:
            token_ids, char_token_ids = self.sent2ids_char(s2, word2id, char2id)
            feed_s2.append(token_ids)
            feed_char_s2.append(char_token_ids)

        feed_dict = {
            self.multi_s1: utils.pad_sequences(feed_multi_s1, padding='post'),
            self.char_multi_s1: utils.pad_sequences(feed_char_multi_s1, padding='post'),
            self.s2: utils.pad_sequences(feed_s2, padding='post'),
            self.char_s2: utils.pad_sequences(feed_char_s2, padding='post'),
        }

        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        if mode == 'infer':
            return feed_dict

        if mode in ['train', 'dev']:
            assert batch_target, 'batch_target should not be None when mode is train or dev'
            feed_dict[self.target] = batch_target
            return feed_dict

        raise ValueError(f'mode type {mode} not support')

    @classmethod
    def generate_data(cls, file, token2id_dct):
        word2id = token2id_dct['word2id']
        char2id = token2id_dct['char2id']
        data = {
            'multi_s1': [],
            's2': [],
            'target': [],
            'char_multi_s1': [],
            'char_s2': [],
        }
        with open(file, 'r', encoding='U8') as f:
            for i, line in enumerate(f):
                item = line.strip().split('\t')
                if len(item) != 3:
                    print('error item:', repr(line))
                    continue
                multi_s1 = item[0].split('$$$')
                s2 = item[1]
                target = item[2]
                multi_s1_ids, char_multi_s1_ids = cls.multi_sent2ids_char(multi_s1, word2id, char2id, max_word_len=50)
                s2_ids, char_s2_ids = cls.sent2ids_char(s2, word2id, char2id, max_word_len=50)
                target_id = int(target)
                if i < 5:  # check
                    print(f'check {i}:')
                    print(f'{multi_s1} -> {multi_s1_ids}')
                    print(f'{s2} -> {s2_ids}')
                    print(f'{target} -> {target_id}')
                    print(f'{multi_s1} -> {char_multi_s1_ids}')
                    print(f'{s2} -> {char_s2_ids}')
                data['multi_s1'].append(multi_s1_ids)
                data['s2'].append(s2_ids)
                data['target'].append(target_id)
                data['char_multi_s1'].append(char_multi_s1_ids)
                data['char_s2'].append(char_s2_ids)
        data['num_data'] = len(data['s2'])
        return data

    @classmethod
    def generate_tfrecord(cls, file, token2id_dct, tfrecord_file):
        from qiznlp.common.tfrecord_utils import items2tfrecord
        word2id = token2id_dct['word2id']
        char2id = token2id_dct['char2id']

        def items_gen():
            with open(file, 'r', encoding='U8') as f:
                for i, line in enumerate(f):
                    item = line.strip().split('\t')
                    if len(item) != 3:
                        print('error item:', repr(line))
                        continue
                    try:
                        multi_s1 = item[0].split('$$$')
                        s2 = item[1]
                        target = item[2]
                        multi_s1_ids, char_multi_s1_ids = cls.multi_sent2ids_char(multi_s1, word2id, char2id, max_word_len=50)
                        s2_ids, char_s2_ids = cls.sent2ids_char(s2, word2id, char2id, max_word_len=50)
                        target_id = int(target)
                        if i < 5:  # check
                            print(f'check {i}:')
                            print(f'{multi_s1} -> {multi_s1_ids}')
                            print(f'{s2} -> {s2_ids}')
                            print(f'{target} -> {target_id}')
                            print(f'{multi_s1} -> {char_multi_s1_ids}')
                            print(f'{s2} -> {char_s2_ids}')
                        d = {
                            'multi_s1': multi_s1_ids,
                            's2': s2_ids,
                            'target': target_id,
                            'char_multi_s1': char_multi_s1_ids,
                            'char_s2': char_s2_ids,
                        }
                        yield d
                    except Exception as e:
                        print('Exception occur in items_gen()!\n', e)
                        continue

        count = items2tfrecord(items_gen(), tfrecord_file)
        return count

    @classmethod
    def load_tfrecord(cls, tfrecord_file, batch_size=128, index=None, shard=None):
        from qiznlp.common.tfrecord_utils import tfrecord2dataset
        if not os.path.exists(tfrecord_file):
            return None, None
        feat_dct = {
            'multi_s1': tf.VarLenFeature(tf.int64),  # [turn,len]
            's2': tf.VarLenFeature(tf.int64),  # [len]
            'target': tf.FixedLenFeature([], tf.int64),
            'char_multi_s1': tf.VarLenFeature(tf.int64),  # [turn,len,char]
            'char_s2': tf.VarLenFeature(tf.int64),  # [len,char]
        }
        shape_dct = {
            'multi_s1': [conf.max_turn, -1],
            'char_multi_s1': [conf.max_turn, -1, conf.max_char],
            'char_s2': [-1, conf.max_char],
        }
        dataset, count = tfrecord2dataset(tfrecord_file, feat_dct, shape_dct=shape_dct, batch_size=batch_size, auto_pad=True, index=index, shard=shard)
        return dataset, count

    def get_signature_export_model(self):
        inputs_dct = {
            'multi_s1': self.multi_s1,
            's2': self.s2,
            'dropout_rate': self.dropout_rate,
            'char_multi_s1': self.char_multi_s1,
            'char_s2': self.char_s2,

        }
        outputs_dct = {
            'y_prob': self.y_prob,
        }
        return inputs_dct, outputs_dct

    @classmethod
    def get_signature_load_pbmodel(cls):
        inputs_lst = ['multi_s1', 's2', 'dropout_rate', 'char_multi_s1', 'char_s2']
        outputs_lst = ['y_prob']
        return inputs_lst, outputs_lst

    @classmethod
    def from_pbmodel(cls, pbmodel_dir, sess):
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], pbmodel_dir)  # 从pb模型载入graph和variable,绑定到sess
        signature = meta_graph_def.signature_def  # 取出signature_def
        default_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY  # or 'chitchat_predict' 签名
        inputs_lst, output_lst = cls.get_signature_load_pbmodel()  # 类方法
        pb_dict = {}
        for k in inputs_lst:
            pb_dict[k] = sess.graph.get_tensor_by_name(signature[default_key].inputs[k].name)  # 从signature中获取输入输出的tensor_name,并从graph中取出
        for k in output_lst:
            pb_dict[k] = sess.graph.get_tensor_by_name(signature[default_key].outputs[k].name)
        model = cls(build_graph=False)  # 里面不再构造图
        for k, v in pb_dict.items():
            setattr(model, k, v)  # 绑定必要的输入输出到实例
        return model

    @classmethod
    def from_ckpt_meta(cls, ckpt_name, sess, graph):
        with graph.as_default():
            saver = tf.train.import_meta_graph(ckpt_name + '.meta', clear_devices=True)
            sess.run(tf.global_variables_initializer())

        model = cls(build_graph=False)  # 里面不再构造图
        # 绑定必要的输入输出到实例
        model.multi_s1 = graph.get_tensor_by_name('multi_s1:0')
        model.s2 = graph.get_tensor_by_name('s2:0')
        model.dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        model.char_multi_s1 = graph.get_tensor_by_name('char_multi_s1:0')
        model.char_s2 = graph.get_tensor_by_name('char_s2:0')
        model.y_prob = graph.get_tensor_by_name('y_prob:0')

        saver.restore(sess, ckpt_name)
        print(f':: restore success! {ckpt_name}')
        return model, saver
