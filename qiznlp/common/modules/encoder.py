#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from tensorflow.python.util import nest
from .beam_search import *
from .embedding import *
from .common_layers import *


def transformer_encoder(encoder_input, encoder_valid_mask,
                        hidden_size=512,
                        filter_size=2048,
                        num_heads=8,
                        num_encoder_layers=6,
                        dropout=0.,
                        attention_dropout=0.,
                        relu_dropout=0.,
                        ):
    # attention mask
    encoder_pad_mask = 1. - encoder_valid_mask  # 1 for pad 0 for nonpad 用于ffn  [batch,len]
    encoder_self_attention_bias = encoder_pad_mask * -1e9  # attention mask
    encoder_self_attention_bias = tf.expand_dims(tf.expand_dims(encoder_self_attention_bias, axis=1), axis=1)  # [batch,1,1,length]

    x = encoder_input
    with tf.variable_scope('encoder'):
        for layer in range(num_encoder_layers):
            with tf.variable_scope('layer_%d' % layer):
                with tf.variable_scope('self_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, dropout_rate=dropout),
                        None,  # self attention
                        encoder_self_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout)
                    x = layer_postprecess(x, y, dropout_rate=dropout)
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        layer_preprocess(x, dropout_rate=dropout),
                        filter_size,
                        hidden_size,
                        relu_dropout=relu_dropout,
                        pad_mask=encoder_pad_mask)
                    x = layer_postprecess(x, y, dropout_rate=dropout)
        encoder_output = layer_preprocess(x, dropout_rate=dropout)
    return encoder_output


def transformer_decoder(decoder_input, encoder_output, decoder_valid_mask, encoder_valid_mask,
                        cache=None,
                        hidden_size=512,
                        filter_size=2048,
                        num_heads=8,
                        num_decoder_layers=6,
                        dropout=0.,
                        attention_dropout=0.,
                        relu_dropout=0.,
                        ):
    # attention mask
    encoder_pad_mask = 1. - encoder_valid_mask  # [batch,length] 1 for pad 0 for nonpad
    encoder_decoder_attention_bias = encoder_pad_mask * -1e9  # attention mask
    encoder_decoder_attention_bias = tf.expand_dims(tf.expand_dims(encoder_decoder_attention_bias, axis=1), axis=1)  # [batch,1,1,length]

    if decoder_valid_mask is None:  # infer mode  decoder_input [batch,1,embed]
        decoder_pad_mask = None
        decoder_self_attention_bias = tf.zeros([1, 1, 1, 1])  # infer阶段其实不需要mask
    else:  # train model
        decoder_pad_mask = 1. - decoder_valid_mask  # [batch,length] 1 for pad; 0 for nonpad 用于ffn
        decoder_length = shape_list(decoder_input)[1]
        decoder_self_attention_bias = attention_bias_lower_triangle(decoder_length)  # 下三角 causal attention

    x = decoder_input
    with tf.variable_scope('decoder'):
        for layer in range(num_decoder_layers):
            layer_name = 'layer_%d' % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope('self_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, dropout_rate=dropout),
                        None,
                        decoder_self_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        cache=layer_cache)
                    x = layer_postprecess(x, y, dropout_rate=dropout)
                with tf.variable_scope('encdec_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, dropout_rate=dropout),
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hidden_size,
                        hidden_size,
                        hidden_size,
                        num_heads,
                        attention_dropout,
                        cache=layer_cache)
                    x = layer_postprecess(x, y, dropout_rate=dropout)
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        layer_preprocess(x, dropout_rate=dropout),
                        filter_size,
                        hidden_size,
                        relu_dropout=relu_dropout,
                        pad_mask=decoder_pad_mask)
                    x = layer_postprecess(x, y, dropout_rate=dropout)
        decoder_output = layer_preprocess(x, dropout_rate=dropout)
    return decoder_output


def attention_base_pool(inputs, input_valid_mask,
                        name='sent_pool', reuse=tf.AUTO_REUSE,
                        attn_v=None):
    """ sentence vector """
    # inputs [batch,len,hidden]
    # input_valid_mask [batch,len]
    # attn_v if exist [hidden]
    hidden_size = shape_list(inputs)[-1]
    if attn_v is None:
        with tf.variable_scope(name, reuse=reuse):
            attn_v = tf.get_variable('attention_v', [hidden_size],  # [hidden]
                                     initializer=tf.contrib.layers.xavier_initializer())

    # attention alpha before normalizer
    score = tf.reduce_sum(inputs * attn_v, axis=-1)  # [batch,len]
    input_pad_mask = 1. - input_valid_mask  # 1 for pad 0 for nonpad
    attention_mask = input_pad_mask * -1e9  # attention mask

    # normalizer
    alignments = tf.nn.softmax(score + attention_mask)  # [batch,len]

    outputs = tf.reduce_sum(inputs * tf.expand_dims(alignments, axis=-1), axis=1)  # [batch,hidden]

    return outputs


def multi_head_attention_base_pool(inputs, input_valid_mask,
                                   total_key_size,
                                   total_value_size,
                                   num_heads,
                                   name='sent_pool',
                                   reuse=tf.AUTO_REUSE,
                                   dropout=0.,
                                   attn_v=None):
    """ sentence vector """
    # inputs [batch,len,hidden]
    # return [batch,hidden]
    hidden_size = shape_list(inputs)[-1]
    if attn_v is None:
        with tf.variable_scope(name, reuse=reuse):
            attn_v = tf.get_variable('attention_v', [hidden_size],  # [hidden]
                                     initializer=tf.contrib.layers.xavier_initializer())

    encoder_pad_mask = 1. - input_valid_mask  # 1 for pad 0 for nonpad [batch,len]
    bias = encoder_pad_mask * -1e9  # attention mask
    bias = tf.expand_dims(tf.expand_dims(bias, axis=1), axis=1)  # [batch,1,1,length]

    attn_v = tf.reshape(attn_v, [1, 1, hidden_size])  # [1,1,hidden]

    # inputs -> k,v  attn_v -> q
    q = tf.layers.dense(attn_v, total_key_size, use_bias=False, name='q')  # [1,1,key_size]
    q = tf.tile(q, [tf.shape(inputs)[0], 1, 1])  # [batch,1,key_size]

    k = tf.layers.dense(inputs, total_key_size, use_bias=False, name='k')  # [batch,len,key_size]
    v = tf.layers.dense(inputs, total_value_size, use_bias=False, name='v')  # [batch,len,key_size]

    # split head
    q = split_heads(q, num_heads)  # [batch,head,1,hidden-]s
    k = split_heads(k, num_heads)  # [batch,head,len,hidden-]
    v = split_heads(v, num_heads)  # [batch,head,len,hidden-]

    key_size_per_head = total_key_size // num_heads
    q = q * key_size_per_head ** -0.5  # scale

    # q * k_T * v
    x = dot_product_attention(q, k, v, bias, dropout_rate=dropout)  # [batch,head,1,hidden]

    x = combine_heads(x)  # [batch,1,hidden]

    x.set_shape(x.shape.as_list()[:-1] + [total_key_size])  # set last dim specifically

    x = tf.squeeze(x, 1)

    x = tf.layers.dense(x, total_value_size, use_bias=False, name='output_transform')  # [batch,hiden]

    return x


def mean_pool(inputs, input_valid_mask, name='mean_pool'):
    # inputs [batch,len,hidden]
    # input_valid_mask [batch,len]  # 1 for nonpad 0 for pad

    with tf.name_scope(name):
        seq_len = tf.reduce_sum(input_valid_mask, axis=-1)  # [batch]

        # set pad vector to zero
        x = inputs * tf.expand_dims(input_valid_mask, -1)  # [batch,len,hidden]

        # mean pool
        x_mean = tf.reduce_sum(x, axis=1) / tf.cast(tf.expand_dims(seq_len, -1), tf.float32)

    return x_mean  # [batch, hidden]


def max_pool(inputs, input_valid_mask, name='max_pool'):
    # inputs [batch,len,hidden]
    # input_valid_mask [batch,len]  # 1 for nonpad 0 for pad

    with tf.name_scope(name):
        # set pad vector to negative infinity
        input_pad_mask = 1. - input_valid_mask
        x = inputs + tf.expand_dims(input_pad_mask * -1e9, -1)  # [batch,len,hidden]

        # max pool
        x_max = tf.reduce_max(x, axis=1)

    return x_max  # [batch, hidden]


def example_transformer_encoder():
    """ transformer encoder example code (just for reference) """
    encoder_input = None  # [batch,len,embed]
    dropout_rate = None
    encoder_valid_mask = mask_nonpad_from_embedding(encoder_input)  # [batch,len] 1 for nonpad; 0 for pad
    encoder_input = add_timing_signal_1d(encoder_input)  # add position embedding
    encoder_input = tf.layers.dropout(encoder_input, dropout_rate)
    encoder_output = transformer_encoder(encoder_input, encoder_valid_mask)


def example_transformer_decoder():
    """ transformer decoder example code (just for reference) """
    decoder_input = None  # [batch,len,embed]
    dropout_rate = None
    encoder_output = None  # [batch,len,hid]
    encoder_valid_mask = None  # [batch,len]

    decoder_valid_mask = mask_nonpad_from_embedding(decoder_input)  # [batch,len] 1 for nonpad; 0 for pad
    decoder_input = shift_right(decoder_input)  # 用pad当做eos
    decoder_input = add_timing_signal_1d(decoder_input)
    decoder_input = tf.layers.dropout(decoder_input, dropout_rate)
    decoder_output = transformer_decoder(decoder_input, encoder_output, decoder_valid_mask, encoder_valid_mask)


def example_transformer_encoder_decoder_train_and_infer():
    """ transformer encoder and decoder example code. especially decoder infer code (just for reference) """
    # encoder
    example_transformer_encoder()

    # decoder
    example_transformer_decoder()

    # transformer decoder infer
    # these args should stay the same with the decoder setting ahead
    encoder_output = None
    encoder_valid_mask = None
    num_decoder_layer = 6
    hidden_size = 512
    num_head = 6
    vocab_size = 10000
    embed_size = 512
    beam_size = 1  # greedy search
    beam_size = 10  # beam search
    max_decode_len = 50
    eos_id = 2

    batch_size = shape_list(encoder_output)[0]
    # 初始化缓存
    cache = {
        'layer_%d' % layer: {
            # 用以缓存decoder过程前面已计算的k,v
            'k': split_heads(tf.zeros([batch_size, 0, hidden_size]), num_head),
            'v': split_heads(tf.zeros([batch_size, 0, hidden_size]), num_head)
        } for layer in range(num_decoder_layer)
    }
    for layer in range(num_decoder_layer):
        # 对于decoder每层均需与encoder顶层隐状态计算attention,相应均有特定的k,v可缓存
        layer_name = 'layer_%d' % layer
        with tf.variable_scope('decoder/%s/encdec_attention/multihead_attention' % layer_name):
            k_encdec = tf.layers.dense(encoder_output, hidden_size, use_bias=False, name='k', reuse=tf.AUTO_REUSE)
            k_encdec = split_heads(k_encdec, num_head)
            v_encdec = tf.layers.dense(encoder_output, hidden_size, use_bias=False, name='v', reuse=tf.AUTO_REUSE)
            v_encdec = split_heads(v_encdec, num_head)
        cache[layer_name]['k_encdec'] = k_encdec
        cache[layer_name]['v_encdec'] = v_encdec
    cache['encoder_output'] = encoder_output
    cache['encoder_mask'] = encoder_valid_mask

    position_embedding = get_timing_signal_1d(max_decode_len, hidden_size)  # position embedding +eos [1,length+1,embed]

    def symbols_to_logits_fn(ids, i, cache):
        ids = ids[:, -1:]  # [batch,1] 截取最后一个
        target_embed, _ = embedding(tf.expand_dims(ids, axis=-1), vocab_size, embed_size, 'embedding')  # [batch,1,hidden] scope name需与前面保持一致

        decoder_input = target_embed + position_embedding[:, i:i + 1, :]  # [batch,1,hidden]

        encoder_output = cache['encoder_output']
        encoder_mask = cache['encoder_mask']

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            decoder_output = transformer_decoder(decoder_input, encoder_output, None, encoder_mask,
                                                 cache=cache,  # 注意infer要cache
                                                 hidden_size=hidden_size,
                                                 filter_size=hidden_size * 4,
                                                 num_heads=num_head,
                                                 num_decoder_layers=num_decoder_layer,
                                                 )
        logits = proj_logits(decoder_output, embed_size, vocab_size, name='embedding')  # [batch,1,vocab]
        ret = tf.squeeze(logits, axis=1)  # [batch,vocab]
        return ret, cache

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

    def greedy_search_wrapper():
        """ Greedy Search """
        decoded_ids, scores = greedy_search(
            symbols_to_logits_fn,
            initial_ids,
            max_decode_len,
            cache=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    def beam_search_wrapper():
        """ Beam Search """
        decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            max_decode_len,
            vocab_size,
            alpha=0,
            states=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    decoded_ids, scores = tf.cond(tf.equal(beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

    # decoded_ids  # [batch,beam/1,len]
    # scores  # [batch,beam/1]


def rnn_decoder_train(decoder_input, encoder_output, decoder_valid_mask, encoder_valid_mask,
                      hidden_size, dropout_rate, num_decoder_layers=1,
                      cell_name='GRUCell', attention_name='BahdanauAttention'):
    """ use tf attention wrapper """
    # encoder_output: [batch,len,hid]
    # encoder_valid_mask: [batch,len]
    # cell_name: GRUCell or LSTMCell
    # attention_name: BahdanauAttention or LuongAttention
    encoder_output_seqlen = tf.cast(tf.reduce_sum(encoder_valid_mask, -1), tf.int32)  # [batch]

    Cell = getattr(tf.nn.rnn_cell, cell_name)  # class GRUCell/LSTMCell
    layers = [tf.nn.rnn_cell.DropoutWrapper(Cell(hidden_size), input_keep_prob=1.0 - dropout_rate) for _ in range(num_decoder_layers)]

    decoder_input = shift_right(decoder_input)  # 用0当做eos
    decoder_input_seqlen = tf.cast(tf.reduce_sum(decoder_valid_mask, -1), tf.int32)  # [batch]

    attention_mechanism = getattr(tf.contrib.seq2seq, attention_name)(
        hidden_size,
        encoder_output,
        memory_sequence_length=encoder_output_seqlen
    )

    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        attention_mechanism
    )

    with tf.variable_scope('decoder'):
        output, _ = tf.nn.dynamic_rnn(
            cell,
            decoder_input,
            decoder_input_seqlen,
            initial_state=None,  # 默认用0向量初始化
            dtype=tf.float32,
            time_major=False
        )  # 默认scope是rnn e.g.decoder/rnn/kernal

    return output


def rnn_decoder_infer(encoder_output, encoder_valid_mask,
                      hidden_size, dropout_rate, beam_size, vocab_size, num_decoder_layers=1,
                      cell_name='GRUCell', attention_name='BahdanauAttention', max_decode_len=50, eos_id=2):
    """ use tf attention wrapper infer """
    # encoder_output: [batch,len,hid]
    # encoder_valid_mask: [batch,len]
    # cell_name: GRUCell or LSTMCell
    # attention_name: BahdanauAttention or LuongAttention

    encoder_output_seqlen = tf.cast(tf.reduce_sum(encoder_valid_mask, -1), tf.int32)  # [batch]

    Cell = getattr(tf.nn.rnn_cell, cell_name)  # class GRUCell/LSTMCell
    layers = [tf.nn.rnn_cell.DropoutWrapper(Cell(hidden_size), input_keep_prob=1.0 - dropout_rate) for _ in range(num_decoder_layers)]

    batch_size = shape_list(encoder_output)[0]

    attention_mechanism = getattr(tf.contrib.seq2seq, attention_name)(
        hidden_size,
        encoder_output,
        memory_sequence_length=encoder_output_seqlen
    )

    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        attention_mechanism,
        output_attention=False
    )

    initial_state = cell.zero_state(batch_size * beam_size, tf.float32)  # 内部会检查batch_size与encoder_output是否一致,需乘beam_size

    # 区分能否设在cache: cache的值在beam_search过程中会expand和merge,需要tensor rank大于1
    cache = {
        'cell_state': initial_state.cell_state,
        'attention': initial_state.attention,
        'alignments': initial_state.alignments,
        'attention_state': initial_state.attention_state,
    }
    unable_cache = {
        'alignment_history': initial_state.alignment_history,
        'time': initial_state.time
    }

    # 将cache先变回batch,beam_search过程会expand/merge/gather,使得state是符合batch*beam的
    cache = nest.map_structure(lambda s: s[:batch_size], cache)

    def symbols_to_logits_fn(ids, i, cache):
        nonlocal unable_cache
        ids = ids[:, -1:]
        targets = tf.expand_dims(ids, axis=-1)  # [batch,1,1]
        embedding_target = embedding(targets, vocab_size, hidden_size, 'shared_embedding')

        decoder_input = tf.squeeze(embedding_target, axis=1)  # [batch,hidden]
        # 合并 cache和unable_cache为state
        state = cell.zero_state(batch_size * beam_size, tf.float32).clone(
            cell_state=cache['cell_state'],
            attention=cache['attention'],
            alignments=cache['alignments'],
            attention_state=cache['attention_state'],
            alignment_history=unable_cache['alignment_history'],
            time=unable_cache['time'],
        )

        with tf.variable_scope('decoder/rnn'):
            output, state = cell(decoder_input, state)
        # 分开cache和unable_cache
        cache['cell_state'] = state.cell_state
        cache['attention'] = state.attention
        cache['alignments'] = state.alignments
        cache['attention_state'] = state.attention_state
        unable_cache['alignment_history'] = state.alignment_history
        unable_cache['time'] = state.time
        body_output = output  # [batch,hidden]

        logits = proj_logits(body_output, hidden_size, vocab_size, name='shared_embedding')
        return logits, cache

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

    def greedy_search_wrapper():
        """ Greedy Search """
        decoded_ids, scores = greedy_search(
            symbols_to_logits_fn,
            initial_ids,
            max_decode_len,
            cache=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    def beam_search_wrapper():
        """ Beam Search """
        decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            max_decode_len,
            vocab_size,
            alpha=0,
            states=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    decoded_ids, scores = tf.cond(tf.equal(beam_size, 1), greedy_search_wrapper, beam_search_wrapper)

    # decoded_ids  # [batch,beam/1,len]
    # scores  # [batch,beam/1]


def example_rnn_decoder_train_infer():
    encoder_output = None
    s1_seqlen = None
    s2 = None
    dropout_rate = 0.
    hidden_size, vocab_size, embed_size = None, None, None
    batch_size = shape_list(encoder_output)[0]
    beam_size = None,
    max_decode_len = 50,
    eos_id = 2
    # decoder
    decoder_rnn = getattr(tf.nn.rnn_cell, 'GRUCell')(hidden_size)  # GRUCell/LSTMCell
    encdec_atten = EncDecAttention(encoder_output, s1_seqlen, hidden_size)

    s2_embed, _ = embedding(tf.expand_dims(s2, -1), vocab_size, embed_size, name='share_embedding')
    s2_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len] 1 for nonpad; 0 for pad
    s2_seqlen = tf.cast(tf.reduce_sum(s2_mask, -1), tf.int32)  # [batch]

    decoder_input = s2_embed
    decoder_input = shift_right(decoder_input)  # 用pad当做eos
    decoder_input = tf.layers.dropout(decoder_input, rate=dropout_rate)  # dropout

    init_decoder_state = decoder_rnn.zero_state(batch_size, tf.float32)
    # init_decoder_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    time_step = tf.constant(0, dtype=tf.int32)
    rnn_output = tf.zeros([batch_size, 0, hidden_size])
    context_output = tf.zeros([batch_size, 0, hidden_size * 2])  # 注意力

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
            tf.TensorShape([None, None, hidden_size]),
            tf.TensorShape([None, None, hidden_size * 2]),
        ])
    # body_output = tf.concat([rnn_output, context_output], axis=-1)
    # body_output = tf.layers.dense(body_output, self.hidden_size, activation=tf.nn.tanh, use_bias=True, name='body_output_layer')
    decoder_output = rnn_output

    logits = proj_logits(decoder_output, embed_size, vocab_size, name='share_embedding')
    # logits = proj_logits(encoder_output[:,:,:300], embed_size, vocab_size, name='share_embedding')

    onehot_s2 = tf.one_hot(s2, depth=vocab_size)  # [batch,len,vocab]

    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_s2)  # [batch,len]
    weights = tf.to_float(tf.not_equal(s2, 0))  # [batch,len] 1 for nonpad; 0 for pad

    loss_num = xentropy * weights  # [batch,len]
    loss_den = weights  # [batch,len]

    loss = tf.reduce_sum(loss_num) / tf.reduce_sum(loss_den)  # scalar

    # decoder infer 
    cache = {'state': decoder_rnn.zero_state(batch_size, tf.float32)}

    def symbols_to_logits_fn(ids, i, cache):
        # ids [batch,length]
        pred_target = ids[:, -1:]  # 截取最后一个  [batch,1]
        embed_target, _ = embedding(tf.expand_dims(pred_target, axis=-1), vocab_size, embed_size, 'share_embedding')  # [batch,length,embed]
        decoder_input = tf.squeeze(embed_target, axis=1)  # [batch,embed]

        # if use attention
        s = cache['state'] if isinstance(decoder_rnn, tf.nn.rnn_cell.GRUCell) else cache['state'].h
        context = encdec_atten(s, beam_size=beam_size)  # [batch,hidden]
        decoder_input = tf.concat([decoder_input, context], axis=-1)  # [batch,hidden+]

        # run rnn
        # with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
        decoder_output, cache['state'] = decoder_rnn(decoder_input, cache['state'])

        logits = proj_logits(decoder_output, hidden_size, vocab_size, name='share_embedding')

        return logits, cache

    initial_ids = tf.zeros([batch_size], dtype=tf.int32)  # <pad>为<sos>

    def greedy_search_wrapper():
        """ Greedy Search """
        decoded_ids, scores = greedy_search(
            symbols_to_logits_fn,
            initial_ids,
            max_decode_len,
            cache=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    def beam_search_wrapper():
        """ Beam Search """
        decoded_ids, scores = beam_search(  # [batch,beam,len] [batch,beam]
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            max_decode_len,
            vocab_size,
            alpha=0,
            states=cache,
            eos_id=eos_id,
        )
        return decoded_ids, scores

    decoded_ids, scores = tf.cond(tf.equal(beam_size, 1), greedy_search_wrapper, beam_search_wrapper)


class EncDecAttention():
    """ encoder decoder Bahdanauattention 可以将key进行缓存 """

    # memory -> key
    # query <-> key -> atten
    # att * memory -> ctx
    def __init__(self, memory, memory_length, attention_size):
        # memory [batch,len,memory_size(hidden)]
        # memory_length [batch] == seq_length
        with tf.variable_scope('encdec_attention', reuse=tf.AUTO_REUSE):
            self.attention_size = attention_size
            self.memory = memory
            self.memory_length = memory_length
            self.processed_key = tf.layers.dense(self.memory, self.attention_size, activation=None, use_bias=False, name='key_layer')  # [batch,len,atten]
            self.atten_v = tf.get_variable('attention_v', shape=[self.attention_size], dtype=self.memory.dtype)

    def __call__(self, query, beam_size=None):
        """ beam_size is not None 代表query的batch扩为batch*beam了,相应key也要扩"""
        if beam_size is not None:  # 支持beam_size = Tensor(1) Tensor不能直接==None
            processed_key_shape = shape_list(self.processed_key)
            processed_key = tf.expand_dims(self.processed_key, axis=1)  # [batch,1,len,hidden]
            processed_key = tf.tile(processed_key, [1, beam_size, 1, 1])  # [batch,beam,len,hidden]
            processed_key = tf.reshape(processed_key, [-1, processed_key_shape[1], processed_key_shape[2]])  # [batch*beam,len,hidden]

            memory_length = tf.expand_dims(self.memory_length, axis=1)  # [batch,1]
            memory_length = tf.tile(memory_length, [1, beam_size])  # [batch,beam]
            memory_length = tf.reshape(memory_length, [-1])  # [batch*beam]
        else:
            processed_key = self.processed_key
            memory_length = self.memory_length

        # query [batch,hidden]
        with tf.variable_scope('encdec_attention', reuse=tf.AUTO_REUSE):
            processed_query = tf.layers.dense(query, self.attention_size, activation=None, use_bias=False, name='query_layer')  # [batch,atten]
        processed_query = tf.expand_dims(processed_query, axis=1)  # [batch,1,atten]

        # attention alpha before normalizer
        score = tf.reduce_sum(self.atten_v * tf.tanh(processed_key + processed_query), axis=2)  # [batch,len] 加性注意力
        # tf.boolean_mask()
        # mask [batch,length]
        score_mask = tf.sequence_mask(memory_length, maxlen=tf.shape(score)[1])  # bool
        score_mask_values = -1e7 * tf.ones_like(score)
        score = tf.where(score_mask, score, score_mask_values)

        # normalizer
        alignments = tf.nn.softmax(score)  # [batch,len]

        # context c
        context = tf.expand_dims(alignments, axis=-1) * self.memory  # [batch,length,memory_size]
        context = tf.reduce_sum(context, axis=1)  # [batch,memory_size]
        return context  # [batch,memory_size]


def multihead_attention_encoder(query, key, key_valid_mask,
                                hidden_size, num_heads, dropout_rate,
                                name='multihead_attention', reuse=tf.AUTO_REUSE,
                                dropout_broadcast_dims=None):
    with tf.variable_scope(name, reuse=reuse):
        key_pad_mask = 1. - key_valid_mask  # 1 for pad 0 for nonpad 用于ffn  [batch,len]
        key_attention_bias = key_pad_mask * -1e9  # attention mask
        key_attention_bias = tf.expand_dims(tf.expand_dims(key_attention_bias, axis=1), axis=1)  # [batch,1,1,len]
        # key_attention_bias [batch,1,1,len_k] + [batch,head,len_q,len_k]
        output = multihead_attention(
            query,
            key,
            key_attention_bias,
            hidden_size,
            hidden_size,
            hidden_size,
            num_heads,
            dropout_rate,
        )
        output += query

        output = layer_norm(output)

        output = dropout_with_broadcast_dims(output, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)

    return output


def conv_multi_kernel(inputs, filters, kernels_size=(1, 2, 3, 4), strides=1, padding='SAME',
                      name='conv_multi_kernel', reuse=tf.AUTO_REUSE,
                      activation=None, use_bias=False,
                      layer_norm_finally=True,
                      ):
    # mrfn function
    # inputs [batch,len,hid]
    with tf.variable_scope(name, reuse=reuse):
        output_lst = []
        for i, kernel_size in enumerate(kernels_size):
            output = tf.layers.conv1d(inputs, filters, kernel_size, strides=strides, padding=padding,
                                      activation=activation, use_bias=use_bias, name='conv1d_%d' % i)
            output_lst.append(output)
        output = tf.concat(output_lst, -1)  # [batch,len,filter*len(kernel)
        if layer_norm_finally:
            output = layer_norm(output)
        return output


def batch_coattention_nnsubmulti(utterance, response, utterance_mask, scope="co_attention", reuse=None):
    # mrfn function
    with tf.variable_scope(scope, reuse=reuse):
        dim = utterance.get_shape().as_list()[-1]
        weight = tf.get_variable('Weight', shape=[dim, dim], dtype=tf.float32)
        e_utterance = tf.einsum('aij,jk->aik', utterance, weight)
        a_matrix = tf.matmul(response, tf.transpose(e_utterance, perm=[0, 2, 1]))
        reponse_atten = tf.matmul(masked_softmax(a_matrix, utterance_mask), utterance)
        feature_mul = tf.multiply(reponse_atten, response)
        feature_sub = tf.subtract(reponse_atten, response)
        feature_last = tf.layers.dense(tf.concat([feature_mul, feature_sub], axis=-1), dim, use_bias=True, activation=tf.nn.relu, reuse=reuse)
    return feature_last


def masked_softmax(scores, mask):
    # mrfn function
    numerator = tf.exp(scores - tf.reduce_max(scores, 2, keepdims=True)) * tf.expand_dims(mask, axis=1)
    denominator = tf.reduce_sum(numerator, 2, keepdims=True)
    weights = tf.div(numerator + 1e-5 / tf.cast(tf.shape(mask)[-1], tf.float32), denominator + 1e-5)
    return weights
