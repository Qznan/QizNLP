import os
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

from qiznlp.common.modules.common_layers import shape_list, mask_nonpad_from_embedding, add_timing_signal_1d, get_timing_signal_1d, shift_right, split_heads
from qiznlp.common.modules.embedding import embedding, proj_logits
from qiznlp.common.modules.encoder import transformer_encoder, transformer_decoder, EncDecAttention
from qiznlp.common.modules.birnn import Bi_RNN, RNN
from qiznlp.common.modules.beam_search import beam_search, greedy_search, get_state_shape_invariants
import qiznlp.common.utils as utils

conf = utils.dict2obj({
    'max_turn': 5,
    'vocab_size': 4000,
    'embed_size': 300,
    'hidden_size': 300,
    'num_heads': 6,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'pretrain_emb': None,
    'beam_size': 5,
    'max_decode_len': 50,
    'eos_id': 2,
})


class Model(object):
    def __init__(self, build_graph=True, **kwargs):
        self.conf = conf
        self.run_model = kwargs.get('run_model', None)  # acquire outside run_model instance
        if build_graph:
            # build placeholder
            self.build_placeholder()
            # build model
            self.model_name = kwargs.get('model_name', 'HRED')
            {
                'HRED': self.build_model1,  # 1507.04808 Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models https://github.com/julianser/hed-dlg 
                'HRAN': self.build_model2,  # 1701.07149 Hierarchical Recurrent Attention Network for Response Generation https://github.com/chenhongshen/HVMN/tree/master/model
                'RECOSA': self.build_model3,  # 1907.05339 ReCoSa Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation https://github.com/zhanghainan/ReCoSa 
                # add new here
            }[self.model_name]()
            print(f'model_name: {self.model_name} build graph ok!')

    def build_placeholder(self):
        # placeholder
        # 原则上模型输入输出不变，不需换新model
        self.multi_s1 = tf.placeholder(tf.int32, [None, None, None], name='multi_s1')  # [batch,turn,len]
        self.s2 = tf.placeholder(tf.int32, [None, None], name='s2')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    def build_model1(self):
        self.uttn_enc_hidden_size = 256
        self.ctx_enc_hidden_size = 256
        batch_size, num_turns, length = shape_list(self.multi_s1)

        # embedding
        # [batch,len,turn,embed]
        multi_s1_embed, _ = embedding(tf.expand_dims(self.multi_s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        # [batch,len,embed]
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)

        # uttn encoder
        uttn_input = tf.reshape(multi_s1_embed, [-1, length, conf.embed_size])  # [batch*turn,len,embed]
        uttn_mask = mask_nonpad_from_embedding(uttn_input)  # [batch*turn,len] 1 for nonpad; 0 for pad
        uttn_seqlen = tf.cast(tf.reduce_sum(uttn_mask, axis=-1), tf.int32)  # [batch*turn]
        # uttn-gru
        self.encoder_uttn_rnn = RNN(cell_name='GRUCell', name='uttn_enc', hidden_size=self.uttn_enc_hidden_size, dropout_rate=self.dropout_rate)
        _, uttn_embed = self.encoder_uttn_rnn(uttn_input, uttn_seqlen)  # [batch*turn,hid]
        uttn_embed = tf.reshape(uttn_embed, [batch_size, num_turns, self.uttn_enc_hidden_size])  # [batch,turn,hid]  # 之后turn相当于len

        # ctx encoder
        ctx_mask = mask_nonpad_from_embedding(uttn_embed)  # [batch,turn] 1 for nonpad; 0 for pad
        ctx_seqlen = tf.cast(tf.reduce_sum(ctx_mask, axis=-1), tf.int32)  # [batch]
        # ctx-gru
        self.encoder_ctx_rnn = RNN(cell_name='GRUCell', name='ctx_enc', hidden_size=self.ctx_enc_hidden_size, dropout_rate=self.dropout_rate)
        _, ctx_embed = self.encoder_ctx_rnn(uttn_embed, ctx_seqlen)  # [batch,hid]

        # rnn decoder train (no attention)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, axis=-1), tf.int32)  # [batch]

        decoder_input = shift_right(s2_embed)  # 用pad当做eos
        decoder_input = tf.layers.dropout(decoder_input, rate=self.dropout_rate)  # dropout

        # 输入拼上ctx
        decoder_ctx = tf.tile(tf.expand_dims(ctx_embed, axis=1), [1, shape_list(decoder_input)[1], 1])  # [batch,len,hid]
        decoder_input = tf.concat([decoder_input, decoder_ctx], axis=2)

        self.decoder_rnn = RNN(cell_name='GRUCell', name='dec', hidden_size=conf.embed_size, dropout_rate=self.dropout_rate)
        decoder_output, decoder_state = self.decoder_rnn(decoder_input, s2_seqlen)

        logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')

        onehot_s2 = tf.one_hot(self.s2, depth=conf.vocab_size)  # [batch,len,vocab]

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
        weights = tf.to_float(tf.not_equal(self.s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

        loss_num = xentropy * weights  # [batch,len]
        loss_den = weights  # [batch,len]

        loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar
        self.loss = loss

        # rnn decoder infer (no attention)
        # 放在cache里面的在后面symbols_to_logits_fn函数中都会变成batch * beam
        cache = {'state': self.decoder_rnn.cell.zero_state(batch_size, tf.float32),  # [batch,hid]
                 'ctx': ctx_embed,  # [batch,hid]
                 }

        def symbols_to_logits_fn(ids, i, cache):
            # ids [batch,length]
            pred_target = ids[:, -1:]  # [batch,1] 截取最后一个
            target_embed, _ = embedding(tf.expand_dims(pred_target, axis=-1), conf.vocab_size, conf.embed_size, 'share_embedding')  # [batch,1,embed]
            decoder_input = tf.squeeze(target_embed, axis=1)  # [batch,embed]

            # 输入加上ctx
            decoder_input = tf.concat([decoder_input, cache['ctx']], axis=-1)

            # run rnn
            decoder_output, cache['state'] = self.decoder_rnn.one_step(decoder_input, cache['state'])

            logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')

            return logits, cache

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

        def greedy_search_wrapper():
            """ Greedy Search """
            decoded_ids, scores = greedy_search(
                symbols_to_logits_fn,
                initial_ids,
                conf.max_decode_len,
                cache=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        def beam_search_wrapper():
            """ Beam Search """
            decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
                symbols_to_logits_fn,
                initial_ids,
                conf.beam_size,
                conf.max_decode_len,
                conf.vocab_size,
                alpha=0,
                states=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        decoded_ids, scores = tf.cond(tf.equal(conf.beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

        self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')  # [batch,beam/1,len]
        self.scores = tf.identity(scores, name='scores')  # [batch,beam/1]

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_model2(self):
        self.uttn_enc_hidden_size = 256
        self.ctx_enc_hidden_size = 256
        batch_size, num_turns, length = shape_list(self.multi_s1)

        # embedding
        # [batch,len,turn,embed]
        multi_s1_embed, _ = embedding(tf.expand_dims(self.multi_s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        # [batch,len,embed]
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)

        # uttn encoder
        uttn_input = tf.reshape(multi_s1_embed, [-1, length, conf.embed_size])  # [batch*turn,len,embed]
        uttn_mask = mask_nonpad_from_embedding(uttn_input)  # [batch*turn,len] 1 for nonpad; 0 for pad
        uttn_seqlen = tf.cast(tf.reduce_sum(uttn_mask, axis=-1), tf.int32)  # [batch*turn]
        # uttn-gru
        self.encoder_uttn_rnn = Bi_RNN(cell_name='GRUCell', name='uttn_enc', hidden_size=self.uttn_enc_hidden_size, dropout_rate=self.dropout_rate)
        uttn_repre, uttn_embed = self.encoder_uttn_rnn(uttn_input, uttn_seqlen)  # [batch*turn,len,2hid] [batch*turn,2hid]
        uttn_embed = tf.reshape(uttn_embed, [batch_size, num_turns, self.uttn_enc_hidden_size * 2])  # [batch,turn,2hid]
        uttn_repre = tf.reshape(uttn_repre, [batch_size, num_turns, length, self.uttn_enc_hidden_size * 2])  # [batch,turn,len,2hid]

        # ctx encoder
        ctx_mask = mask_nonpad_from_embedding(uttn_embed)  # [batch,turn] 1 for nonpad; 0 for pad
        ctx_seqlen = tf.cast(tf.reduce_sum(ctx_mask, axis=-1), tf.int32)  # [batch]

        # reverse turn
        uttn_repre = tf.reverse_sequence(uttn_repre, seq_lengths=ctx_seqlen, seq_axis=1, batch_axis=0)

        # ctx-gru
        self.encoder_ctx_rnn = RNN(cell_name='GRUCell', name='ctx_enc', hidden_size=self.ctx_enc_hidden_size, dropout_rate=self.dropout_rate)
        init_ctx_encoder_state = self.encoder_ctx_rnn.cell.zero_state(batch_size, tf.float32)

        # rnn decoder train
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, axis=-1), tf.int32)  # [batch]

        decoder_input = shift_right(s2_embed)  # 用pad当做eos
        decoder_input = tf.layers.dropout(decoder_input, rate=self.dropout_rate)  # dropout

        self.decoder_rnn = RNN(cell_name='GRUCell', name='dec', hidden_size=conf.embed_size, dropout_rate=self.dropout_rate)
        init_decoder_state = self.decoder_rnn.cell.zero_state(batch_size, tf.float32)

        # 两重循环 -_-
        # 外层循环是decoder解码的每一步 time_step
        def loop_condition(time_step, *_):
            return tf.less(time_step, tf.reduce_max(s2_seqlen))

        def loop_body(time_step, dec_rnn_state, dec_rnn_output):
            # cal decoder ctx
            # word level attention

            # 内层循环是对于解码每一步 ctx_rnn上turn的每一步，最终生成ctx序列 turn_step
            def inner_loop_condition(turn_step, *_):
                return tf.less(turn_step, tf.reduce_max(ctx_seqlen))

            def inner_loop_body(turn_step, ctx_rnn_state, ctx_rnn_output):
                # 根据si, ctx_init_state 递归计算多个turn的句子ctx
                q_antecedent = tf.concat([ctx_rnn_state, dec_rnn_state], axis=-1)  # [batch, h]
                q_antecedent = tf.tile(tf.expand_dims(q_antecedent, 1), [1, length, 1])  # [batch,len,h]

                # 抽取每个batch的第i个turn
                # sent_repre [batch,turn,len,h]
                q_antecedent = tf.concat([uttn_repre[:, turn_step, :, :], q_antecedent], -1)  # [batch,len,h] 
                uttn_mask_in_turn = tf.reshape(uttn_mask, [batch_size, num_turns, length])[:, turn_step, :]  # [batch,len]

                # word-level-attn
                h = tf.layers.dense(q_antecedent, 128, activation=tf.nn.tanh, use_bias=True, name='word_level_attn/layer1')
                energy = tf.layers.dense(h, 1, use_bias=True, name='word_level_attn/layer2')  # [batch,len,1]
                energy = tf.squeeze(energy, -1) + (1. - uttn_mask_in_turn) * -1e9
                alpha = tf.nn.softmax(energy)  # [batch,len]
                r_in_turn = tf.reduce_sum(tf.expand_dims(alpha, -1) * uttn_repre[:, turn_step, :, :], 1)  # [batch,h]

                ctx_rnn_output_, ctx_rnn_state = self.encoder_ctx_rnn.one_step(r_in_turn, ctx_rnn_state)

                # attch
                ctx_rnn_output = tf.concat([ctx_rnn_output, tf.expand_dims(ctx_rnn_output_, 1)], 1)  # [batch,turn,h]
                return turn_step + 1, ctx_rnn_state, ctx_rnn_output

            # start inner loop
            final_turn_step, final_state, ctx_rnn_output = tf.while_loop(
                inner_loop_condition,
                inner_loop_body,
                loop_vars=[tf.constant(0, dtype=tf.int32),
                           init_ctx_encoder_state,
                           tf.zeros([batch_size, 0, self.ctx_enc_hidden_size])],
                shape_invariants=[
                    tf.TensorShape([]),
                    nest.map_structure(get_state_shape_invariants, init_ctx_encoder_state),
                    tf.TensorShape([None, None, self.ctx_enc_hidden_size]),
                ])

            # ctx_rnn_output  # [batch,turn,h]
            # dec_rnn_state  # [batch,h]
            # ctx-level-attn
            # q_antecedent = tf.tile(tf.expand_dims(dec_rnn_state, axis=1), [1, num_turns, 1])  # [batch,turn,h]
            # 这样只拿当前batch中的尽可能小的turns数量而不是固定turn
            q_antecedent = tf.tile(tf.expand_dims(dec_rnn_state, axis=1), [1, shape_list(ctx_rnn_output)[1], 1])  # [batch,turn,h]
            q_antecedent = tf.concat([q_antecedent, ctx_rnn_output], 2)  # [batch,turn,h]
            h = tf.layers.dense(q_antecedent, 128, activation=tf.nn.tanh, use_bias=True, name='ctx_level_attn/layer1')
            energy = tf.layers.dense(h, 1, use_bias=True, name='ctx_level_attn/layer2')  # [batch,turn,1]
            energy = tf.squeeze(energy, -1) + (1. - ctx_mask) * -1e9  # [batch,turn]
            alpha = tf.nn.softmax(energy)  # [batch,turn]
            ctx_input_in_dec = tf.reduce_sum(tf.expand_dims(alpha, -1) * ctx_rnn_output, 1)  # [batch,h]

            dec_rnn_input = tf.concat([ctx_input_in_dec, decoder_input[:, time_step, :]], -1)  # [batch,h]
            dec_rnn_output_, dec_rnn_state = self.decoder_rnn.one_step(dec_rnn_input, dec_rnn_state)

            dec_rnn_output = tf.concat([dec_rnn_output, tf.expand_dims(dec_rnn_output_, 1)], 1)

            return time_step + 1, dec_rnn_state, dec_rnn_output

        # start outer loop
        final_time_step, final_state, dec_rnn_output = tf.while_loop(
            loop_condition,
            loop_body,
            loop_vars=[tf.constant(0, dtype=tf.int32),
                       init_decoder_state,
                       tf.zeros([batch_size, 0, conf.embed_size])],
            shape_invariants=[
                tf.TensorShape([]),
                nest.map_structure(get_state_shape_invariants, init_decoder_state),
                tf.TensorShape([None, None, conf.embed_size]),
            ])

        decoder_output = dec_rnn_output

        logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')

        onehot_s2 = tf.one_hot(self.s2, depth=conf.vocab_size)  # [batch,len,vocab]

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
        weights = tf.to_float(tf.not_equal(self.s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

        loss_num = xentropy * weights  # [batch,len]
        loss_den = weights  # [batch,len]

        loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar
        self.loss = loss

        # rnn decoder infer
        # 放在cache里面的在后面symbols_to_logits_fn函数中都会变成batch * beam
        cache = {'dec_rnn_state': self.decoder_rnn.cell.zero_state(batch_size, tf.float32),  # [batch,hid]
                 'ctx_rnn_state': self.encoder_ctx_rnn.cell.zero_state(batch_size, tf.float32),  # [batch,hid]
                 'uttn_repre': uttn_repre  # [batch,turn,len,2hid]
                 }

        def symbols_to_logits_fn(ids, i, cache):
            # ids [batch,length]
            pred_target = ids[:, -1:]  # [batch,1] 截取最后一个
            target_embed, _ = embedding(tf.expand_dims(pred_target, axis=-1), conf.vocab_size, conf.embed_size, 'share_embedding')  # [batch,1,embed]
            decoder_input = tf.squeeze(target_embed, axis=1)  # [batch,embed]

            dec_rnn_state = cache['dec_rnn_state']

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                # 内层循环是对于解码每一步 ctx_rnn上turn的每一步，最终生成ctx序列 turn_step
                def inner_loop_condition(turn_step, *_):
                    return tf.less(turn_step, tf.reduce_max(ctx_seqlen))

                def inner_loop_body(turn_step, ctx_rnn_state, ctx_rnn_output):
                    # 根据si, ctx_init_state 递归计算多个turn的句子ctx
                    q_antecedent = tf.concat([ctx_rnn_state, dec_rnn_state], axis=-1)  # [batch, h]
                    q_antecedent = tf.tile(tf.expand_dims(q_antecedent, 1), [1, length, 1])  # [batch,len,h]

                    # 抽取每个batch的第i个turn
                    # sent_repre [batch,turn,len,h]
                    q_antecedent = tf.concat([cache['uttn_repre'][:, turn_step, :, :], q_antecedent], -1)  # [batch,len,h] 
                    uttn_mask_in_turn = tf.reshape(uttn_mask, [batch_size, num_turns, length])[:, turn_step, :]  # [batch,len]

                    # word-level-attn
                    h = tf.layers.dense(q_antecedent, 128, activation=tf.nn.tanh, use_bias=True, name='word_level_attn/layer1')
                    energy = tf.layers.dense(h, 1, use_bias=True, name='word_level_attn/layer2')  # [batch,len,1]
                    energy = tf.squeeze(energy, -1) + (1. - uttn_mask_in_turn) * -1e9
                    alpha = tf.nn.softmax(energy)  # [batch,len]
                    r_in_turn = tf.reduce_sum(tf.expand_dims(alpha, -1) * cache['uttn_repre'][:, turn_step, :, :], 1)  # [batch,h]

                    ctx_rnn_output_, ctx_rnn_state = self.encoder_ctx_rnn.one_step(r_in_turn, ctx_rnn_state)
                    # attch
                    ctx_rnn_output = tf.concat([ctx_rnn_output, tf.expand_dims(ctx_rnn_output_, 1)], 1)  # [batch,turn,h]
                    return turn_step + 1, ctx_rnn_state, ctx_rnn_output

                # start inner loop
                final_turn_step, final_state, ctx_rnn_output = tf.while_loop(
                    inner_loop_condition,
                    inner_loop_body,
                    loop_vars=[tf.constant(0, dtype=tf.int32),
                               cache['ctx_rnn_state'],
                               tf.zeros([shape_list(cache['ctx_rnn_state'])[0], 0, self.ctx_enc_hidden_size])],
                    shape_invariants=[
                        tf.TensorShape([]),
                        nest.map_structure(get_state_shape_invariants, init_ctx_encoder_state),
                        tf.TensorShape([None, None, self.ctx_enc_hidden_size]),
                    ])

                # ctx_rnn_output  # [batch,turn,h]
                # dec_rnn_state  # [batch,h]
                # ctx-level-attn
                # q_antecedent = tf.tile(tf.expand_dims(dec_rnn_state, axis=1), [1, num_turns, 1])  # [batch,turn,h]
                # 这样只拿当前batch中的尽可能小的turns数量而不是固定turn
                q_antecedent = tf.tile(tf.expand_dims(dec_rnn_state, axis=1), [1, shape_list(ctx_rnn_output)[1], 1])  # [batch,turn,h]
                q_antecedent = tf.concat([q_antecedent, ctx_rnn_output], 2)  # [batch,turn,h]
                h = tf.layers.dense(q_antecedent, 128, activation=tf.nn.tanh, use_bias=True, name='ctx_level_attn/layer1')
                energy = tf.layers.dense(h, 1, use_bias=True, name='ctx_level_attn/layer2')  # [batch,turn,1]
                energy = tf.squeeze(energy, -1) + (1. - ctx_mask) * -1e9  # [batch,turn]
                alpha = tf.nn.softmax(energy)  # [batch,turn]
                ctx_input_in_dec = tf.reduce_sum(tf.expand_dims(alpha, -1) * ctx_rnn_output, 1)  # [batch,h]

                dec_rnn_input = tf.concat([ctx_input_in_dec, decoder_input], -1)  # [batch,h]
                dec_rnn_output_, dec_rnn_state = self.decoder_rnn.one_step(dec_rnn_input, dec_rnn_state)

                cache['dec_rnn_state'] = dec_rnn_state

                logits = proj_logits(dec_rnn_output_, conf.embed_size, conf.vocab_size, name='share_embedding')

                return logits, cache

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

        def greedy_search_wrapper():
            """ Greedy Search """
            decoded_ids, scores = greedy_search(
                symbols_to_logits_fn,
                initial_ids,
                conf.max_decode_len,
                cache=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        def beam_search_wrapper():
            """ Beam Search """
            decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
                symbols_to_logits_fn,
                initial_ids,
                conf.beam_size,
                conf.max_decode_len,
                conf.vocab_size,
                alpha=0,
                states=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        decoded_ids, scores = tf.cond(tf.equal(conf.beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

        self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')  # [batch,beam/1,len]
        self.scores = tf.identity(scores, name='scores')  # [batch,beam/1]

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        
    def build_model3(self):
        assert not conf.hidden_size % 2  # 被2整除
        self.uttn_enc_hidden_size = conf.hidden_size // 2
        batch_size, num_turns, length = shape_list(self.multi_s1)

        # embedding
        # [batch,len,turn,embed]
        multi_s1_embed, _ = embedding(tf.expand_dims(self.multi_s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        # [batch,len,embed]
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)

        # uttn encoder
        uttn_input = tf.reshape(multi_s1_embed, [-1, length, conf.embed_size])  # [batch*turn,len,embed]
        uttn_mask = mask_nonpad_from_embedding(uttn_input)  # [batch*turn,len] 1 for nonpad; 0 for pad
        uttn_seqlen = tf.cast(tf.reduce_sum(uttn_mask, axis=-1), tf.int32)  # [batch*turn]
        # uttn-gru
        self.encoder_uttn_rnn = Bi_RNN(cell_name='GRUCell', name='uttn_enc', hidden_size=self.uttn_enc_hidden_size, dropout_rate=self.dropout_rate)
        _, uttn_embed = self.encoder_uttn_rnn(uttn_input, uttn_seqlen)  # [batch*turn,len,2hid] [batch*turn,2hid]
        uttn_embed = tf.reshape(uttn_embed, [batch_size, num_turns, self.uttn_enc_hidden_size * 2])  # [batch,turn,2hid]  # 之后turn相当于len

        # transformer ctx encoder
        encoder_valid_mask = mask_nonpad_from_embedding(uttn_embed)  # [batch,turn] 1 for nonpad; 0 for pad
        encoder_input = add_timing_signal_1d(uttn_embed)  # add position embedding
        encoder_input = tf.layers.dropout(encoder_input, rate=self.dropout_rate)  # dropout

        encoder_output = transformer_encoder(encoder_input, encoder_valid_mask,
                                             hidden_size=conf.hidden_size,
                                             filter_size=conf.hidden_size * 4,
                                             num_heads=conf.num_heads,
                                             num_encoder_layers=conf.num_encoder_layers,
                                             dropout=self.dropout_rate,
                                             attention_dropout=self.dropout_rate,
                                             relu_dropout=self.dropout_rate,
                                             )

        # transformer decoder
        decoder_input = s2_embed
        decoder_valid_mask = mask_nonpad_from_embedding(decoder_input)  # [batch,len] 1 for nonpad; 0 for pad
        decoder_input = shift_right(decoder_input)  # 用pad当做eos
        decoder_input = add_timing_signal_1d(decoder_input)
        decoder_input = tf.layers.dropout(decoder_input, rate=self.dropout_rate)  # dropout

        decoder_output = transformer_decoder(decoder_input, encoder_output, decoder_valid_mask, encoder_valid_mask,
                                             cache=None,
                                             hidden_size=conf.hidden_size,
                                             filter_size=conf.hidden_size * 4,
                                             num_heads=conf.num_heads,
                                             num_decoder_layers=conf.num_decoder_layers,
                                             dropout=self.dropout_rate,
                                             attention_dropout=self.dropout_rate,
                                             relu_dropout=self.dropout_rate,
                                             )

        logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')

        onehot_s2 = tf.one_hot(self.s2, depth=conf.vocab_size)  # [batch,len,vocab]

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
        weights = tf.to_float(tf.not_equal(self.s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

        loss_num = xentropy * weights  # [batch,len]
        loss_den = weights  # [batch,len]

        loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar
        self.loss = loss

        # transformer decoder infer
        # 放在cache里面的在后面symbols_to_logits_fn函数中都会变成batch * beam
        # 初始化缓存
        cache = {
            'layer_%d' % layer: {
                # 用以缓存decoder过程前面已计算的k,v
                'k': split_heads(tf.zeros([batch_size, 0, conf.embed_size]), conf.num_heads),
                'v': split_heads(tf.zeros([batch_size, 0, conf.embed_size]), conf.num_heads)
            } for layer in range(conf.num_decoder_layers)
        }
        for layer in range(conf.num_decoder_layers):
            # 对于decoder每层均需与encoder顶层隐状态计算attention,相应均有特定的k,v可缓存
            layer_name = 'layer_%d' % layer
            with tf.variable_scope('decoder/%s/encdec_attention/multihead_attention' % layer_name):
                k_encdec = tf.layers.dense(encoder_output, conf.embed_size, use_bias=False, name='k', reuse=tf.AUTO_REUSE)
                k_encdec = split_heads(k_encdec, conf.num_heads)
                v_encdec = tf.layers.dense(encoder_output, conf.embed_size, use_bias=False, name='v', reuse=tf.AUTO_REUSE)
                v_encdec = split_heads(v_encdec, conf.num_heads)
            cache[layer_name]['k_encdec'] = k_encdec
            cache[layer_name]['v_encdec'] = v_encdec
        cache['encoder_output'] = encoder_output
        cache['encoder_mask'] = encoder_valid_mask

        # position embedding
        position_embedding = get_timing_signal_1d(conf.max_decode_len, conf.embed_size)  # +eos [1,length+1,embed]

        def symbols_to_logits_fn(ids, i, cache):
            ids = ids[:, -1:]  # [batch,1] 截取最后一个
            target_embed, _ = embedding(tf.expand_dims(ids, axis=-1), conf.vocab_size, conf.embed_size, 'share_embedding', reuse=True)  # [batch,1,hidden]

            decoder_input = target_embed + position_embedding[:, i:i + 1, :]  # [batch,1,hidden]

            encoder_output = cache['encoder_output']
            encoder_mask = cache['encoder_mask']

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                decoder_output = transformer_decoder(decoder_input, encoder_output, None, encoder_mask,
                                                     cache=cache,  # 注意infer要cache
                                                     hidden_size=conf.embed_size,
                                                     filter_size=conf.embed_size * 4,
                                                     num_heads=6,
                                                     num_decoder_layers=6,
                                                     )
            logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')  # [batch,1,vocab]
            ret = tf.squeeze(logits, axis=1)  # [batch,vocab]
            return ret, cache

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

        def greedy_search_wrapper():
            """ Greedy Search """
            decoded_ids, scores = greedy_search(
                symbols_to_logits_fn,
                initial_ids,
                conf.max_decode_len,
                cache=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        def beam_search_wrapper():
            """ Beam Search """
            decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
                symbols_to_logits_fn,
                initial_ids,
                conf.beam_size,
                conf.max_decode_len,
                conf.vocab_size,
                alpha=0,
                states=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        decoded_ids, scores = tf.cond(tf.equal(conf.beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

        self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')  # [batch,beam/1,len]
        self.scores = tf.identity(scores, name='scores')  # [batch,beam/1]

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
        # 至少需要固定一个(turn或len),且需要补齐为矩阵turn * len,后面才可通过reshape恢复
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

    def create_feed_dict_from_data(self, data, ids, mode='train'):
        # data:数据已经转为id, data不同字段保存该段字段全量数据
        batch_multi_s1 = [data['multi_s1'][i] for i in ids]
        batch_s2 = [data['s2'][i] for i in ids]

        # 多轮必定需要补齐，就不做判断了
        batch_multi_s1 = utils.pad_sequences(batch_multi_s1, padding='post')

        if len(set([len(e) for e in batch_s2])) != 1:  # 长度不等
            batch_s2 = utils.pad_sequences(batch_s2, padding='post')
        feed_dict = {
            self.multi_s1: batch_multi_s1,
            self.s2: batch_s2,
        }
        if mode == 'train': feed_dict['num'] = len(batch_s2)
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_features(self, features, mode='train'):
        # feature:tfrecord数据的example, 每个features的不同字段包括该字段一个batch数据
        feed_dict = {
            self.multi_s1: features['multi_s1'],
            self.s2: features['s2'],
        }
        if mode == 'train': feed_dict['num'] = len(features['s2'])
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_raw(self, batch_multi_s1, batch_s2, token2id_dct, mode='infer'):
        word2id = token2id_dct['word2id']

        feed_multi_s1 = [self.multi_sent2ids(multi_s1.split('$$$'), word2id) for multi_s1 in batch_multi_s1]

        feed_dict = {
            self.multi_s1: utils.pad_sequences(feed_multi_s1, padding='post'),
        }
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        if mode == 'infer':
            return feed_dict

        if mode in ['train', 'dev']:
            assert batch_s2, 'batch_s2 should not be None when mode is train or dev'
            feed_s2 = [self.sent2ids(s2, word2id) for s2 in batch_s2]
            feed_dict[self.s2] = utils.pad_sequences(feed_s2, padding='post')
            return feed_dict

        raise ValueError(f'mode type {mode} not support')

    @classmethod
    def generate_data(cls, file, token2id_dct):
        word2id = token2id_dct['word2id']
        data = {
            'multi_s1': [],
            's2': [],
        }
        with open(file, 'r', encoding='U8') as f:
            for i, line in enumerate(f):
                item = line.strip().split('\t')
                if len(item) != 2:
                    print('error item:', repr(line))
                    continue
                multi_s1 = item[0].split('$$$')
                s2 = item[1]
                multi_s1_ids = cls.multi_sent2ids(multi_s1, word2id, max_word_len=50)
                s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                if i < 5:  # check
                    print(f'check {i}:')
                    print(f'{multi_s1} -> {multi_s1_ids}')
                    print(f'{s2} -> {s2_ids}')
                data['multi_s1'].append(multi_s1_ids)
                data['s2'].append(s2_ids)
        data['num_data'] = len(data['s2'])
        return data

    @classmethod
    def generate_tfrecord(cls, file, token2id_dct, tfrecord_file):
        from qiznlp.common.tfrecord_utils import items2tfrecord
        word2id = token2id_dct['word2id']

        def items_gen():
            with open(file, 'r', encoding='U8') as f:
                for i, line in enumerate(f):
                    item = line.strip().split('\t')
                    if len(item) != 2:
                        print('error item:', repr(line))
                        continue
                    try:
                        multi_s1 = item[0].split('$$$')
                        s2 = item[1]
                        multi_s1_ids = cls.multi_sent2ids(multi_s1, word2id, max_word_len=50)
                        s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                        if i < 5:  # check
                            print(f'check {i}:')
                            print(f'{multi_s1} -> {multi_s1_ids}')
                            print(f'{s2} -> {s2_ids}')
                        d = {
                            'multi_s1': multi_s1_ids,
                            's2': s2_ids,
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
            'multi_s1': tf.VarLenFeature(tf.int64),
            's2': tf.VarLenFeature(tf.int64),
        }
        shape_dct = {
            'multi_s1': [conf.max_turn, -1],
        }
        dataset, count = tfrecord2dataset(tfrecord_file, feat_dct, shape_dct=shape_dct, batch_size=batch_size, auto_pad=True, index=index, shard=shard)
        return dataset, count

    def get_signature_export_model(self):
        inputs_dct = {
            'multi_s1': self.multi_s1,
            'dropout_rate': self.dropout_rate,
        }
        outputs_dct = {
            'decoded_ids': self.decoded_ids,
            'scores': self.scores,
        }
        return inputs_dct, outputs_dct

    @classmethod
    def get_signature_load_pbmodel(cls):
        inputs_lst = ['multi_s1', 'dropout_rate']
        outputs_lst = ['decoded_ids', 'scores']
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
        model.dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        model.decoded_ids = graph.get_tensor_by_name('decoded_ids:0')
        model.scores = graph.get_tensor_by_name('scores:0')

        saver.restore(sess, ckpt_name)
        print(f':: restore success! {ckpt_name}')
        return model, saver
