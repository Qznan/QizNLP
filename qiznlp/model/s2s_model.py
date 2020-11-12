import os
import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

from qiznlp.common.modules.common_layers import shape_list, mask_nonpad_from_embedding, add_timing_signal_1d, get_timing_signal_1d, shift_right, split_heads
from qiznlp.common.modules.embedding import embedding, proj_logits
from qiznlp.common.modules.encoder import transformer_encoder, transformer_decoder, EncDecAttention
from qiznlp.common.modules.birnn import Bi_RNN
from qiznlp.common.modules.beam_search import beam_search, greedy_search, get_state_shape_invariants
import qiznlp.common.utils as utils

conf = utils.dict2obj({
    'vocab_size': 4000,
    'embed_size': 300,
    'hidden_size': 300,
    'num_heads': 6,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dropout_rate': 0.2,
    'lr': 1e-3,
    'pretrain_emb': None,
    'beam_size': 40,
    'max_decode_len': 50,
    'eos_id': 2,
    'gamma': 1,  # 多样性鼓励因子
    'num_group': 1,  # 分组beam_search
    'top_k': 30  # 详见beam_search
})


class Model(object):
    def __init__(self, build_graph=True, **kwargs):
        self.conf = conf
        self.run_model = kwargs.get('run_model', None)  # acquire outside run_model instance
        if build_graph:
            # build placeholder
            self.build_placeholder()
            # build model
            self.model_name = kwargs.get('model_name', 'trans')
            {
                'trans': self.build_model1,
                # 'rnn_s2s': self.build_model2,
                'rnn_s2s': self.build_model3,
                # build_model2和build_model3都是rnn_attn_s2s, 只是实现代码不同，前者使用谷歌attn模块，后者独立实现attn以方便适配beam_search。建议使用后者
                # add new here
            }[self.model_name]()
            print(f'model_name: {self.model_name} build graph ok!')

    def build_placeholder(self):
        # placeholder
        # 原则上模型输入输出不变，不需换新model
        self.s1 = tf.placeholder(tf.int32, [None, None], name='s1')
        self.s2 = tf.placeholder(tf.int32, [None, None], name='s2')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    def build_model1(self):
        # embedding  # [batch,len,embed]
        s1_embed, _ = embedding(tf.expand_dims(self.s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)

        # encoder
        encoder_input = s1_embed
        encoder_valid_mask = mask_nonpad_from_embedding(encoder_input)  # [batch,len] 1 for nonpad; 0 for pad
        encoder_input = add_timing_signal_1d(encoder_input)  # add position embedding
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

        # decoder
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
        # self.sent_loss = tf.reduce_sum(loss_num, -1) / tf.reduce_sum(loss_den, -1)  # [batch]

        """ 
        transformer decoder infer 
        """
        batch_size = shape_list(encoder_output)[0]
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
                                                     num_heads=conf.num_heads,
                                                     num_decoder_layers=conf.num_decoder_layers,
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
                max_decode_len=conf.max_decode_len,
                cache=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        def beam_search_wrapper():
            """ Beam Search """
            decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
                symbols_to_logits_fn,
                initial_ids,
                beam_size=conf.beam_size,
                max_decode_len=conf.max_decode_len,
                vocab_size=conf.vocab_size,
                states=cache,
                eos_id=conf.eos_id,
                gamma=conf.gamma,
                num_group=conf.num_group,
                top_k=conf.top_k,
            )
            return decoded_ids, scores

        decoded_ids, scores = tf.cond(tf.equal(conf.beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

        self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')  # [batch,beam/1,len]
        self.scores = tf.identity(scores, name='scores')  # [batch,beam/1]

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_model2(self):
        # biGRU encoder + bah_attn + GRU decoder
        # embedding
        # [batch,len,embed]
        # pretrained_word_embeddings = np.load(f'{curr_dir}/pretrain_emb_300.npy')
        pretrained_word_embeddings = None
        s1_embed, _ = embedding(tf.expand_dims(self.s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=pretrained_word_embeddings)
        s1_mask = mask_nonpad_from_embedding(s1_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s1_seqlen = tf.cast(tf.reduce_sum(s1_mask, axis=-1), tf.int32)  # [batch]

        # encoder
        encoder_input = s1_embed
        encoder_input = tf.layers.dropout(encoder_input, rate=self.dropout_rate)  # dropout

        with tf.variable_scope('birnn_encoder'):
            self.bilstm_encoder1 = Bi_RNN(cell_name='GRUCell', hidden_size=conf.hidden_size, dropout_rate=self.dropout_rate)
            encoder_output, _ = self.bilstm_encoder1(encoder_input, s1_seqlen)  # [batch,len,2hid]

        # decoder

        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=pretrained_word_embeddings)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]

        decoder_input = s2_embed
        decoder_input = shift_right(decoder_input)  # 用pad当做eos
        decoder_input = tf.layers.dropout(decoder_input, rate=self.dropout_rate)  # dropout

        decoder_rnn = tf.nn.rnn_cell.DropoutWrapper(getattr(tf.nn.rnn_cell, 'GRUCell')(conf.hidden_size),  # GRUCell/LSTMCell
                                                    input_keep_prob=1.0 - self.dropout_rate)

        attention_mechanism = getattr(tf.contrib.seq2seq, 'BahdanauAttention')(
            conf.hidden_size,
            encoder_output,
            memory_sequence_length=s1_seqlen,
            name='BahdanauAttention',
        )
        cell = tf.contrib.seq2seq.AttentionWrapper(decoder_rnn,
                                                   attention_mechanism,
                                                   output_attention=False,
                                                   name='attention_wrapper',
                                                   )

        with tf.variable_scope('decoder'):
            decoder_output, _ = tf.nn.dynamic_rnn(
                cell,
                decoder_input,
                s2_seqlen,
                initial_state=None,  # 默认用0向量初始化
                dtype=tf.float32,
                time_major=False
            )  # 默认scope是rnn e.g.decoder/rnn/kernal

        logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')
        # logits = proj_logits(encoder_output[:,:,:300], conf.embed_size, conf.vocab_size, name='share_embedding')

        onehot_s2 = tf.one_hot(self.s2, depth=conf.vocab_size)  # [batch,len,vocab]

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
        weights = tf.to_float(tf.not_equal(self.s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

        loss_num = xentropy * weights  # [batch,len]
        loss_den = weights  # [batch,len]

        loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar
        self.loss = loss
        # self.sent_loss = tf.reduce_sum(loss_num, -1) / tf.reduce_sum(loss_den, -1)  # [batch]

        """ 
        decoder infer 
        """
        # batch_size = shape_list(encoder_output)[0]
        batch_size = shape_list(encoder_output)[0]
        last_dim = shape_list(encoder_output)[-1]
        tile_encoder_output = tf.tile(tf.expand_dims(encoder_output, 1), [1, conf.beam_size, 1, 1])
        tile_encoder_output = tf.reshape(tile_encoder_output, [batch_size * conf.beam_size, -1, last_dim])
        tile_s1_seqlen = tf.tile(tf.expand_dims(s1_seqlen, 1), [1, conf.beam_size])
        tile_s1_seqlent = tf.reshape(tile_s1_seqlen, [-1])

        # 因为tf.BahdanauAttention在初始时就要指定beam_size来tile memory, 所以这里初始化另外一个专用于推断的beam_size tiled的并共享参数
        # 不过验证还是有问题，不能用reuse=True 报 Variable memory_layer_1/kernel does not exist, or was not created with tf.get_variable()
        # 故不建议使用
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            attention_mechanism_decoder = getattr(tf.contrib.seq2seq, 'BahdanauAttention')(
                conf.hidden_size,
                tile_encoder_output,
                memory_sequence_length=tile_s1_seqlent,
                name='BahdanauAttention',
            )
            cell_decoder = tf.contrib.seq2seq.AttentionWrapper(decoder_rnn,
                                                               attention_mechanism_decoder,
                                                               output_attention=False,
                                                               name='attention_wrapper',
                                                               )

        initial_state = cell_decoder.zero_state(batch_size * conf.beam_size, tf.float32)  # 内部会检查batch_size与encoder_output是否一致,需乘beam_size

        # 初始化缓存
        # 区分能否设在cache: cache的值在beam_search过程中会expand和merge,需要tensor rank大于1
        cache = {
            'cell_state': initial_state.cell_state,
            'attention': initial_state.attention,
            'alignments': initial_state.alignments,
            'attention_state': initial_state.attention_state,
        }
        unable_cache = {
            'alignment_history': initial_state.alignment_history,
            # 'time': initial_state.time
        }

        # 将cache先变回batch,beam_search过程会expand/merge/gather,使得state是符合batch*beam的
        cache = nest.map_structure(lambda s: s[:batch_size], cache)

        def symbols_to_logits_fn(ids, i, cache):
            nonlocal unable_cache
            ids = ids[:, -1:]
            target = tf.expand_dims(ids, axis=-1)  # [batch,1,1]
            embedding_target, _ = embedding(target, conf.vocab_size, conf.hidden_size, 'share_embedding', reuse=True)
            input = tf.squeeze(embedding_target, axis=1)  # [batch,hid]

            # 合并 cache和unable_cache为state
            state = cell_decoder.zero_state(batch_size * conf.beam_size, tf.float32).clone(
                cell_state=cache['cell_state'],
                attention=cache['attention'],
                alignments=cache['alignments'],
                attention_state=cache['attention_state'],
                alignment_history=unable_cache['alignment_history'],
                # time=unable_cache['time'],
                time=tf.convert_to_tensor(i, dtype=tf.int32),
            )

            with tf.variable_scope('decoder/rnn', reuse=tf.AUTO_REUSE):
                output, state = cell_decoder(input, state)
            # 分开cache和unable_cache
            cache['cell_state'] = state.cell_state
            cache['attention'] = state.attention
            cache['alignments'] = state.alignments
            cache['attention_state'] = state.attention_state
            unable_cache['alignment_history'] = state.alignment_history
            # unable_cache['time'] = state.time
            body_output = output  # [batch,hidden]

            logits = proj_logits(body_output, conf.embed_size, conf.vocab_size, name='share_embedding')
            return logits, cache

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

        def greedy_search_wrapper():
            """ Greedy Search """
            decoded_ids, scores = greedy_search(
                symbols_to_logits_fn,
                initial_ids,
                max_decode_len=conf.max_decode_len,
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
                states=cache,
                eos_id=conf.eos_id,
                gamma=conf.gamma,
                num_group=conf.num_group,
                top_k=conf.top_k,
            )
            return decoded_ids, scores

        decoded_ids, scores = tf.cond(tf.equal(conf.beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

        self.decoded_ids = tf.identity(decoded_ids, name='decoded_ids')  # [batch,beam/1,len]
        self.scores = tf.identity(scores, name='scores')  # [batch,beam/1]

        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=conf.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def build_model3(self):
        # biGRU encoder + bah_attn + GRU decoder
        # embedding
        # [batch,len,embed]
        # pretrained_word_embeddings = np.load(f'{curr_dir}/pretrain_emb_300.npy')
        pretrained_word_embeddings = None
        s1_embed, _ = embedding(tf.expand_dims(self.s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=pretrained_word_embeddings)
        s1_mask = mask_nonpad_from_embedding(s1_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s1_seqlen = tf.cast(tf.reduce_sum(s1_mask, axis=-1), tf.int32)  # [batch]

        # encoder
        encoder_input = s1_embed
        encoder_input = tf.layers.dropout(encoder_input, rate=self.dropout_rate)  # dropout

        with tf.variable_scope('birnn_encoder'):
            self.bilstm_encoder1 = Bi_RNN(cell_name='GRUCell', hidden_size=conf.hidden_size, dropout_rate=self.dropout_rate)
            encoder_output, _ = self.bilstm_encoder1(encoder_input, s1_seqlen)  # [batch,len,2hid]

        batch_size = shape_list(encoder_input)[0]

        # decoder
        decoder_rnn = getattr(tf.nn.rnn_cell, 'GRUCell')(conf.hidden_size)  # GRUCell/LSTMCell
        encdec_atten = EncDecAttention(encoder_output, s1_seqlen, conf.hidden_size)

        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=pretrained_word_embeddings)
        s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
        s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]

        decoder_input = s2_embed
        decoder_input = shift_right(decoder_input)  # 用pad当做eos
        decoder_input = tf.layers.dropout(decoder_input, rate=self.dropout_rate)  # dropout

        init_decoder_state = decoder_rnn.zero_state(batch_size, tf.float32)
        # init_decoder_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        time_step = tf.constant(0, dtype=tf.int32)
        rnn_output = tf.zeros([batch_size, 0, conf.hidden_size])
        context_output = tf.zeros([batch_size, 0, conf.hidden_size * 2])  # 注意力

        def loop_condition(time_step, *_):
            return tf.less(time_step, tf.reduce_max(s2_seqlen))

        def loop_body(time_step, prev_rnn_state, rnn_output, context_output):
            # attention
            s = prev_rnn_state if isinstance(decoder_rnn, tf.nn.rnn_cell.GRUCell) else prev_rnn_state.h
            context = encdec_atten(s)  # [batch,hidden]
            context_output = tf.concat([context_output, tf.expand_dims(context, axis=1)], axis=1)

            # construct rnn input
            rnn_input = tf.concat([decoder_input[:, time_step, :], context], axis=-1)  # [batch,hidden+]  use attention
            # rnn_input = decoder_input[:, time_step, :]  # [batch,hidden]  not use attention

            # run rnn
            current_output, rnn_state = decoder_rnn(rnn_input, prev_rnn_state)

            # append to output bucket via length dim
            rnn_output = tf.concat([rnn_output, tf.expand_dims(current_output, axis=1)], axis=1)

            return time_step + 1, rnn_state, rnn_output, context_output

        # start loop
        final_time_step, final_state, rnn_output, context_output = tf.while_loop(
            loop_condition,
            loop_body,
            loop_vars=[time_step, init_decoder_state, rnn_output, context_output],
            shape_invariants=[
                tf.TensorShape([]),
                nest.map_structure(get_state_shape_invariants, init_decoder_state),
                tf.TensorShape([None, None, conf.hidden_size]),
                tf.TensorShape([None, None, conf.hidden_size * 2]),
            ])
        # body_output = tf.concat([rnn_output, context_output], axis=-1)
        # body_output = tf.layers.dense(body_output, self.hidden_size, activation=tf.nn.tanh, use_bias=True, name='body_output_layer')
        decoder_output = rnn_output

        logits = proj_logits(decoder_output, conf.embed_size, conf.vocab_size, name='share_embedding')
        # logits = proj_logits(encoder_output[:,:,:300], conf.embed_size, conf.vocab_size, name='share_embedding')

        onehot_s2 = tf.one_hot(self.s2, depth=conf.vocab_size)  # [batch,len,vocab]

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
        weights = tf.to_float(tf.not_equal(self.s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

        loss_num = xentropy * weights  # [batch,len]
        loss_den = weights  # [batch,len]

        loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar
        self.loss = loss
        # self.sent_loss = tf.reduce_sum(loss_num, -1) / tf.reduce_sum(loss_den, -1)  # [batch]


        # decoder infer
        cache = {'state': decoder_rnn.zero_state(batch_size, tf.float32)}
        def symbols_to_logits_fn(ids, i, cache):
            # ids [batch,length]
            pred_target = ids[:, -1:]  # 截取最后一个  [batch,1]
            embed_target, _ = embedding(tf.expand_dims(pred_target, axis=-1), conf.vocab_size, conf.embed_size, 'share_embedding')  # [batch,length,embed]
            decoder_input = tf.squeeze(embed_target, axis=1)  # [batch,embed]

            # if use attention
            s = cache['state'] if isinstance(decoder_rnn, tf.nn.rnn_cell.GRUCell) else cache['state'].h
            context = encdec_atten(s, beam_size=conf.beam_size)  # [batch,hidden]
            decoder_input = tf.concat([decoder_input, context], axis=-1)  # [batch,hidden+]

            # run rnn
            # with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            decoder_output, cache['state'] = decoder_rnn(decoder_input, cache['state'])

            logits = proj_logits(decoder_output, conf.hidden_size, conf.vocab_size, name='share_embedding')

            return logits, cache

        initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

        def greedy_search_wrapper():
            """ Greedy Search """
            decoded_ids, scores = greedy_search(
                symbols_to_logits_fn,
                initial_ids,
                max_decode_len=conf.max_decode_len,
                cache=cache,
                eos_id=conf.eos_id,
            )
            return decoded_ids, scores

        def beam_search_wrapper():
            """ Beam Search """
            decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
                symbols_to_logits_fn,
                initial_ids,
                beam_size=conf.beam_size,
                max_decode_len=conf.max_decode_len,
                vocab_size=conf.vocab_size,
                states=cache,
                eos_id=conf.eos_id,
                gamma=conf.gamma,
                num_group=conf.num_group,
                top_k=conf.top_k,
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
        # sent 已分好词 ' '隔开
        # 形成batch时才动态补齐长度
        words = sent.split(' ')
        token_ids = [word2id.get(word, word2id['<unk>']) for word in words]
        if max_word_len:
            token_ids = token_ids[:max_word_len - 1]
        token_ids.append(word2id['<eos>'])
        return token_ids  # [len]

    def create_feed_dict_from_data(self, data, ids, mode='train'):
        # data:数据已经转为id, data不同字段保存该段字段全量数据
        batch_s1 = [data['s1'][i] for i in ids]
        batch_s2 = [data['s2'][i] for i in ids]
        feed_dict = {
            self.s1: utils.pad_sequences(batch_s1, padding='post'),
            self.s2: utils.pad_sequences(batch_s2, padding='post'),
        }
        if mode == 'train': feed_dict['num'] = len(batch_s1)
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_features(self, features, mode='train'):
        # feature:tfrecord数据的example, 每个features的不同字段包括该字段一个batch数据
        feed_dict = {
            self.s1: features['s1'],
            self.s2: features['s2'],
        }
        if mode == 'train': feed_dict['num'] = len(features['s1'])
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_raw(self, batch_s1, batch_s2, token2id_dct, mode='infer'):
        word2id = token2id_dct['word2id']

        feed_s1 = [self.sent2ids(s1, word2id) for s1 in batch_s1]

        feed_dict = {
            self.s1: utils.pad_sequences(feed_s1, padding='post'),
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
            's1': [],
            's2': [],
        }
        with open(file, 'r', encoding='U8') as f:
            for i, line in enumerate(f):
                item = line.strip().split('\t')
                if len(item) != 2:
                    print('error item:', repr(line))
                    continue
                s1 = item[0]
                s2 = item[1]
                s1_ids = cls.sent2ids(s1, word2id, max_word_len=50)
                s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                if i < 5:  # check
                    print(f'check {i}:')
                    print(f'{s1} -> {s1_ids}')
                    print(f'{s2} -> {s2_ids}')
                data['s1'].append(s1_ids)
                data['s2'].append(s2_ids)
        data['num_data'] = len(data['s1'])
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
                        s1 = item[0]
                        s2 = item[1]
                        s1_ids = cls.sent2ids(s1, word2id, max_word_len=50)
                        s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                        if i < 5:  # check
                            print(f'check {i}:')
                            print(f'{s1} -> {s1_ids}')
                            print(f'{s2} -> {s2_ids}')
                        d = {
                            's1': s1_ids,
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
            # 's1': tf.FixedLenFeature([50], tf.int64),
            's1': tf.VarLenFeature(tf.int64),
            's2': tf.VarLenFeature(tf.int64),
        }
        dataset, count = tfrecord2dataset(tfrecord_file, feat_dct, batch_size=batch_size, auto_pad=True, index=index, shard=shard)
        return dataset, count

    def get_signature_export_model(self):
        inputs_dct = {
            's1': self.s1,
            'dropout_rate': self.dropout_rate,
        }
        outputs_dct = {
            'decoded_ids': self.decoded_ids,
            'scores': self.scores,
        }
        return inputs_dct, outputs_dct

    @classmethod
    def get_signature_load_pbmodel(cls):
        inputs_lst = ['s1', 'dropout_rate']
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
        model.s1 = graph.get_tensor_by_name('s1:0')
        # model.s2 = graph.get_tensor_by_name('s2:0')
        model.dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        model.decoded_ids = graph.get_tensor_by_name('decoded_ids:0')
        model.scores = graph.get_tensor_by_name('scores:0')

        saver.restore(sess, ckpt_name)
        print(f':: restore success! {ckpt_name}')
        return model, saver
