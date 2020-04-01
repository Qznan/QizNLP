#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tensorflow as tf


def mask_nonzero(labels):
    """ mask: Assign weight 1.0(true)
        nonmask: Assign weight 0.0(false)
        mask if value is not 0.0
    """
    return tf.to_float(tf.not_equal(labels, 0))


def mask_nonpad_from_ids(ids):
    """ ids: [batch,length]
        Assign 1.0(true) for non pad, 0.0(false) for pad(id=0, all emb element is 0)
        return [batch,length]
    """
    return tf.to_float(tf.not_equal(ids, 0))


def mask_nonpad_from_embedding(emb):
    """ emb: [batch,length,embed]
        Assign 1.0(true) for non pad, 0.0(false) for pad(id=0, all emb element is 0)
        return [batch,length]
    """
    return mask_nonzero(tf.reduce_sum(tf.abs(emb), axis=-1))


def length_from_embedding(emb):
    """ emb: [batch,length,embed]
        return [batch]
    """
    length = tf.reduce_sum(mask_nonpad_from_embedding(emb), axis=-1)
    length = tf.cast(length, tf.int32)
    return length


def length_from_ids(ids):
    """ ids: [batch,length,1]
        return [batch]
    """
    weight_ids = mask_nonzero(ids)
    length = tf.reduce_sum(weight_ids, axis=[1, 2])
    length = tf.cast(length, tf.int32)
    return length


def shift_right(x, pad_value=None):
    """ shift the second dim of x right by one
        decoder中对target进行右偏一位作为decode_input
        x [batch,length,embed]
        pad_value [tile_batch,1,pad_embed] or
        pad_value [pad_embed]
    """
    if pad_value is not None:
        if len(shape_list(x)) == 1:
            pad_embed = tf.reshape([1, 1, -1])
            pad_embed = tf.tile(pad_embed, [shape_list(x)[0], 1, 1])
            shifted_x = tf.concat([pad_embed, x], axis=1)[:, :-1, :]  # length维度左边补pad_embed
        else:
            shifted_x = tf.concat([pad_value, x], axis=1)[:, :-1, :]  # length维度左边补pad_embed
    else:
        shifted_x = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # length维度左边补pad_embed [0,0...]
    return shifted_x


def shape_list(x):
    """ return list of dims, statically where possible """
    x = tf.convert_to_tensor(x)
    # if unknown rank, return dynamic shape 如果秩都不知道
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = shape[i] if static[i] is None else static[i]
        ret.append(dim)
    return ret


def cast_like(x, y):
    """ cast x to y's dtype, if necessary """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x
    else:
        return tf.cast(x, y.dtype)


def log_prob_from_logits(logits, axis=-1):
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=True)


def sample_with_temperature(logits, temperature):
    """ 0.0:argmax 1.0:sampling >1.0:random """
    # logits [batch,length,vocab]
    # ret [batch,length]
    if temperature == 0.0:
        return tf.argmax(logits, axis=-1)
    else:
        assert temperature > 0.0
        logits_shape = shape_list(logits)
        reshape_logits = tf.reshape(logits, [-1, logits_shape[-1]]) / temperature
        choices = tf.multinomial(reshape_logits, 1)  # 仅采样1个 该方式只支持2-D
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices


def batch_normalization(x, training, name):
    # with tf.variable_scope(name, reuse=)
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(training, lambda: bn_train, lambda: bn_inference)
    return z


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """封装dropout函数,broadcast_dims对应dropout的noise_shape"""
    assert 'noise_shape' not in kwargs
    if broadcast_dims:
        x_shape = tf.shape(x)
        ndims = len(x.get_shape())
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]  # allow dim like -1 as well
        kwargs['noise_shape'] = [1 if i in broadcast_dims else x_shape[i] for i in range(ndims)]  # 类似[1,length,hidden]
    return tf.nn.dropout(x, keep_prob, **kwargs)


def dropout_no_scaling(x, keep_prob):
    """ 不进行放缩的drop, 用以在token上 """
    if keep_prob == 1.0:
        return x
    mask = tf.less(tf.random_uniform(tf.shape(x)), keep_prob)
    mask = cast_like(mask, x)
    return x * mask


def layer_norm(x, epsilon=1e-06):
    """ layer norm """
    filters = shape_list(x)[-1]
    with tf.variable_scope('layer_norm', values=[x], reuse=None):
        scale = tf.get_variable('layer_norm_scale', [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable('layer_norm_bias', [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]  # 写法独特

    mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # mu
    variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)  # sigma
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)  # (x-mu)*sigma
    ret = norm_x * scale + bias
    return ret


def layer_prepostprocess(previous_value, x, sequence, dropout_rate, dropout_broadcast_dims=None):
    """ apply a sequence of function to the input or output of a layer
        a: add previous_value
        n: apply normalization
        d: apply dropout
        for example, if sequence=='dna', then the output is: previous_value + normalize(dropout(x))
    """
    with tf.variable_scope('layer_prepostprocess'):
        if sequence == 'none':
            return x
        for c in sequence:
            if c == 'a':
                x += previous_value  # residual
            elif c == 'n':
                x = layer_norm(x)  # LN
            else:
                assert c == 'd', 'unknown sequence step %s' % c
                x = dropout_with_broadcast_dims(x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)  # dropout
        return x


def layer_preprocess(x, dropout_rate=0., layer_preprocess_sequence='n'):
    assert 'a' not in layer_preprocess_sequence, 'no residual connections allowed in preprocess sequence'
    return layer_prepostprocess(None, x, sequence=layer_preprocess_sequence,
                                dropout_rate=dropout_rate)


def layer_postprecess(previous_value, x, dropout_rate=0., layer_postprocess_sequence='da'):
    return layer_prepostprocess(previous_value, x, sequence=layer_postprocess_sequence,
                                dropout_rate=dropout_rate)


def split_heads(x, num_heads):
    """ x [batch,length,hidden]
        ret [batch,num_heads,length,hidden/num_heads]
    """
    x_shape = shape_list(x)
    last_dim = x_shape[-1]
    if isinstance(last_dim, int) and isinstance(num_heads, int):
        assert last_dim % num_heads == 0
    x = tf.reshape(x, x_shape[:-1] + [num_heads, last_dim // num_heads])
    x = tf.transpose(x, [0, 2, 1, 3])
    return x


def combine_heads(x):
    """ Inverse of split_heads
        x [batch,num_heads,length,hidden/num_head]
        ret [batch,length,hidden]
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    x = tf.reshape(x, x_shape[:-2] + [a * b])
    return x


def compute_qkv(query_antecedent, memory_antecedent, total_key_depth, total_value_depth):
    """total_depth 包括了所有head的depth"""
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = tf.layers.dense(query_antecedent, total_key_depth, use_bias=False, name='q')
    k = tf.layers.dense(memory_antecedent, total_key_depth, use_bias=False, name='k')
    v = tf.layers.dense(memory_antecedent, total_value_depth, use_bias=False, name='v')
    return q, k, v


def dot_product_attention(q, k, v, bias, dropout_rate=0.0, dropout_broadcast_dims=None, name='dot_product_attention'):
    """ """
    with tf.variable_scope(name, values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)  # [batch,num_heads,length_q,length_kv]
        if bias is not None:
            bias = cast_like(bias, logits)
        logits += bias
        weights = tf.nn.softmax(logits, name='attention_weight')  # [batch,num_heads,length_q,length_kv]
        if dropout_rate != 0:
            # drop out attention links for each head
            weights = dropout_with_broadcast_dims(weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        # v [batch,num_heads,length_kv,hidden/num_heads]
        ret = tf.matmul(weights, v)  # [batch,num_heads,length_q,hidden/num_heads]
        return ret


def multihead_attention(query_antecedent, memory_antecedent, bias, total_key_depth, total_value_depth, output_depth,
                        num_heads, dropout_rate, cache=None, name='multihead_attention', dropout_broadcast_dims=None):
    """ """
    assert total_key_depth % num_heads == 0
    assert total_value_depth % num_heads == 0
    with tf.variable_scope(name, values=[query_antecedent, memory_antecedent]):
        if cache is None or memory_antecedent is None:  # training or self_attention in inferring
            q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth, total_value_depth)
        if cache is not None:  # inferring时有cache, 此时query_antecedent均为[batch,1,hidden]
            assert bias is not None, 'Bias required for caching'

            if memory_antecedent is not None:  # encode-decode attention 使用cache
                q = tf.layers.dense(query_antecedent, total_key_depth, use_bias=False, name='q')
                k = cache['k_encdec']
                v = cache['v_encdec']
            else:  # decode self_attention 得到k,v需存到cache
                k = split_heads(k, num_heads)  # [batch,num_heads,length,hidden/num_heads]
                v = split_heads(v, num_heads)
                k = cache['k'] = tf.concat([cache['k'], k], axis=2)
                v = cache['v'] = tf.concat([cache['v'], v], axis=2)
        q = split_heads(q, num_heads)
        if cache is None:
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q = q * key_depth_per_head ** -0.5  # scale

        x = dot_product_attention(q, k, v, bias, dropout_rate, dropout_broadcast_dims=dropout_broadcast_dims)

        x = combine_heads(x)

        x.set_shape(x.shape.as_list()[:-1] + [total_key_depth])  # set last dim specifically

        x = tf.layers.dense(x, output_depth, use_bias=False, name='output_transform')
        return x


def attention_bias_lower_triangle(length):
    """ 下三角矩阵 """
    band = tf.matrix_band_part(tf.ones([length, length]), -1, 0)  # [length,length] 下三角矩阵,下三角均为1,上三角均为0
    # [[1,0,0],
    #  [1,1,0],
    #  [1,1,1]] float
    band = tf.reshape(band, [1, 1, length, length])
    band = -1e9 * (1.0 - band)
    # [[0,-1e9,-1e9],
    #  [0,0,-1e9],
    #  [0,0,0]]
    return band


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """ use for position embedding """
    length = shape_list(x)[1]
    hidden_size = shape_list(x)[2]
    signal = get_timing_signal_1d(length, hidden_size, min_timescale, max_timescale, start_index)
    return x + signal


def get_timing_signal_1d(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """ use for calculate position embedding """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = hidden_size // 2
    log_timescales_increment = np.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescales_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal  # [1,len,hidden]


def transformer_ffn_layer(x, filter_size, hidden_size, relu_dropout=0., pad_mask=None):
    original_shape = shape_list(x)  # [batch,length,hidden]

    if pad_mask is not None:
        """ remove pad """
        flat_pad_mask = tf.reshape(pad_mask, [-1])  # [batch*length]
        flat_nonpad_ids = tf.to_int32(tf.where(tf.equal(flat_pad_mask, 0)))

        # flat x
        x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))  # [batch*length,hidden]
        # remove pad
        x = tf.gather_nd(x, flat_nonpad_ids)  # [batch*length-,hidden]

    h = tf.layers.dense(x, filter_size, use_bias=True, activation=tf.nn.relu, name='conv1')

    if relu_dropout != 0.:
        h = dropout_with_broadcast_dims(h, 1.0 - relu_dropout, broadcast_dims=None)

    o = tf.layers.dense(h, hidden_size, activation=None, use_bias=True, name='conv2')

    if pad_mask is not None:
        """ restore pad """
        o = tf.scatter_nd(  # 将updates中对应的值填充到indices指定的索引中，空的位置会用0代替，刚好代表pad
            indices=flat_nonpad_ids,
            updates=o,
            shape=tf.concat([tf.shape(flat_pad_mask)[:1], tf.shape(o)[1:]], axis=0)
        )

        o = tf.reshape(o, original_shape)
    return o


def sigmoid_cross_entropy(logits, labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=cast_like(labels, logits))
    loss = tf.reduce_mean(loss)
    return loss


def softmax_cross_entropy(logits, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                      labels=cast_like(labels, logits))
    loss = tf.reduce_mean(loss)
    return loss


def am_softmax_loss(logits, labels, margin=0.35, scale=30.):
    labels = tf.cast(labels, tf.float32)
    logits = labels * (logits - margin) + (1 - labels) * logits
    logits *= scale
    loss = softmax_cross_entropy(logits, labels)
    return loss


def margin_loss(logits, labels):
    # logits = tf.nn.softmax(logits)
    labels = tf.cast(labels, tf.float32)
    loss = labels * tf.square(tf.maximum(0., 0.9 - logits)) + \
           0.25 * (1.0 - labels) * tf.square(tf.maximum(0., logits - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss


def l1_loss(logits, labels):
    return tf.reduce_mean(tf.abs(logits - labels))


def l2_loss(logits, labels):
    # tf.nn.l2_loss()?
    return tf.reduce_mean(tf.square(logits - labels))


def hinge_loss(neg_logits, pos_logits, margin, is_distance):
    if is_distance:
        loss = tf.reduce_mean(tf.maximum(margin + pos_logits - neg_logits, 0.0))
    else:
        loss = tf.reduce_mean(tf.maximum(margin - pos_logits + neg_logits, 0.0))
    return loss


def improved_triplet_loss(neg_logits, pos_logits, margin, margin_pos, is_distance):
    if is_distance:
        loss = tf.reduce_mean(tf.maximum(margin + pos_logits - neg_logits, 0.0) +
                              tf.square(1 - neg_logits))
        # tf.square(pos_logits))
    else:
        loss = tf.reduce_mean(tf.maximum(margin - pos_logits + neg_logits, 0.0) +
                              tf.square(neg_logits))
        # tf.square(1 - pos_logits))
    return loss


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      < negative samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    # alpha: the weight(unimportant) of negative
    # positive lable 1   * 1-alpha  0.75  loss更大了，更需要优化
    # negative lable 0   * alpha    0.25  loss更低了，更不需要优化
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    L = -labels * (1 - alpha) * ((1 - y_pred) ** gamma) * tf.log(y_pred) - \
        (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return L


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[-1])
    L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    L = tf.reduce_sum(L, axis=-1)
    return L


def cosine(vec1, vec2):
    # [-1, 1] greater is similar
    vec1_norm = tf.sqrt(tf.reduce_sum(tf.square(vec1), axis=-1))
    vec2_norm = tf.sqrt(tf.reduce_sum(tf.square(vec1), axis=-1))
    dot_product = tf.reduce_sum(tf.multiply(vec1, vec2), axis=-1)
    return dot_product / (vec1_norm * vec2_norm)


def dot_product(vec1, vec2, scale=False):
    # [-inf, inf] greater is similar
    dot = tf.reduce_sum(tf.multiply(vec1, vec2), axis=-1)
    if scale:
        dot = dot / tf.shpae(vec1)[-1] ** -0.5
    return dot


def euclidean(vec1, vec2):
    # [0, inf] smaller is similar
    return tf.sqrt(tf.reduce_sum(tf.square(vec1 - vec2), -1))


def l2_distance(vec1, vec2):
    # [0, inf] smaller is similar
    return tf.reduce_sum(tf.square(vec1 - vec2), -1)


def arccosine(vec1, vec2):
    # 余弦夹角
    # [0, 3.1415] smaller is similar
    return tf.math.acos(cosine(vec1, vec2))


def contrastive_loss(vec1, vec2, labels, metrics='euclidean', margin=1.25):
    # 对比损失 正样本对越相似 负样本对越不相似且达到阈值margin以上
    # vec [batch,hid]
    # labels [batch]
    # labels value:1/0  1 for pos, 0 for neg
    assert metrics in ['euclidean', 'arccosine', 'cosine', 'dot_product'], f'not support metrics {metrics}'

    labels = tf.cast(labels, tf.float32)

    if metrics in ['euclidean', 'arccosine']:
        if metrics == 'euclidean':
            distance = euclidean(vec1, vec2)  # [batch]
        else:
            distance = arccosine(vec1, vec2)  # [batch]
        loss = labels * tf.square(distance) + (1 - labels) * tf.square(tf.maximum(margin - distance), 0.)
        loss = tf.reduce_mean(loss, axis=0)

    if metrics in ['cosine', 'dot_product']:
        if metrics == 'cosine':
            similarity = cosine(vec1, vec2)  # [batch]
        else:
            similarity = dot_product(vec1, vec2, scale=True)  # [batch]
        loss = (1 - labels) * tf.square(similarity) + labels * tf.square(tf.maximum(margin - similarity), 0.)
        loss = tf.reduce_mean(loss, axis=0)

    return loss


def triplet_loss(anchor_vec, pos_vec, neg_vec, metrics='euclidean', margin=1.25):
    # vec [batch,hid]
    # 三元损失 正样本与锚样本相似要大于负样本与锚样本，且达到阈值margin以上
    assert metrics in ['euclidean', 'arccosine', 'cosine', 'dot_product'], f'not support metrics {metrics}'
    if metrics == 'euclidean':
        pos_pair = l2_distance(anchor_vec, pos_vec)
        neg_pair = l2_distance(anchor_vec, neg_vec)
    elif metrics == 'arccosine':
        pos_pair = arccosine(anchor_vec, pos_vec)
        neg_pair = arccosine(anchor_vec, neg_vec)
    elif metrics == 'cosine':
        pos_pair = cosine(anchor_vec, pos_vec)
        neg_pair = cosine(anchor_vec, neg_vec)
    else:
        pos_pair = dot_product(anchor_vec, pos_vec)
        neg_pair = dot_product(anchor_vec, neg_vec)

    if metrics in ['euclidean', 'arccosine']:
        loss = tf.maximum(pos_pair - neg_pair + margin, 0.)  # [batch]
    else:
        loss = tf.maximum(neg_pair - pos_pair + margin, 0.)  # [batch]
    loss = tf.reduce_mean(loss, axis=0)
    return loss


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def center_loss_v3(features, labels, alpha, name='center_loss'):
    # features [batch,hid]
    # label one_hot [batch,num_class]
    hidden_size = features.get_shape().as_list()[-1]
    num_classes = labels.get_shape().as_list()[-1]
    labels = tf.argmax(labels, axis=-1)  # [batch]
    with tf.variable_scope(name):
        centers = tf.get_variable('centers', [num_classes, hidden_size],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=False)
        centers_batch = tf.gather(centers, labels)  # 获取当前batch对应的类中心特征
        c_loss = tf.reduce_mean(tf.nn.l2_loss(features - centers_batch), axis=-1)

        centers_diff = alpha * (centers_batch - features)  # 类中心的梯度
        centers = tf.scatter_sub(centers, labels, centers_diff)  # 更新梯度
    return c_loss, centers


def softplus(x, name="softplus"):
    with tf.variable_scope(name):
        return tf.nn.softplus


def swish(x, name="swish"):
    """f(x) = sigmoid(x) * x
    """
    with tf.variable_scope(name):
        return (tf.nn.sigmoid(x * 1.0) * x)


def leaky_relu(x, leak=0.2, name="leaky_relu"):
    """f(x) = max(alpha * x, x)
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


def cube(x, name="cube_act"):
    """f(x) = pow(x, 3)
    """
    with tf.variable_scope(name):
        return tf.pow(x, 3)


def penalized_tanh(x, name="penalized_tanh"):
    """f(x) = max(tanh(x), alpha * tanh(x))
    """
    with tf.variable_scope(name):
        alpha = 0.25
        return tf.maximum(tf.tanh(x), alpha * tf.tanh(x))


def cosper(x, name="cosper_act"):
    """f(x) = cos(x) - x
    """
    with tf.variable_scope(name):
        return (tf.cos(x) - x)


def minsin(x, name="minsin_act"):
    """f(x) = min(x, xin(x))
    """
    with tf.variable_scope(name):
        return tf.minimum(x, tf.sin(x))


def tanhrev(x, name="tanhprev"):
    """f(x) = pow(atan(x), 2) - x
    """
    with tf.variable_scope(name):
        return (tf.pow(tf.atan(x), 2) - x)


def maxsig(x, name="maxsig_act"):
    """f(x) = max(x, tf.sigmiod(x))
    """
    with tf.variable_scope(name):
        return tf.maximum(x, tf.sigmoid(x))


def maxtanh(x, name="max_tanh_act"):
    """f(x) = max(x, tanh(x))
    """
    with tf.variable_scope(name):
        return tf.maximum(x, tf.tanh(x))


def get_activation(active_type='swish', **kwargs):
    mp = {
        "sigmoid": tf.nn.sigmoid,
        "tanh": tf.nn.tanh,
        "softsign": tf.nn.softsign,
        "relu": tf.nn.relu,
        "leaky_relu": leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "swish": swish,
        "sin": tf.sin,
        "cube": cube,
        "penalized_tanh": penalized_tanh,
        "cosper": cosper,
        "minsin": minsin,
        "tanhrev": tanhrev,
        "maxsig": maxsig,
        "maxtanh": maxtanh,
        "softplus": tf.nn.softplus,
    }
    assert active_type in mp, "%s is not in activation list"
    return mp[active_type]


def cosine_proximity_tf(y_true, y_pred):
    """cosine loss"""
    y_true = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    return -(tf.reduce_sum(y_true * y_pred, axis=-1) + 1) / 2 + 1


def get_state_shape_invariants(tensor):
    """ return the shape of tensor but set middle dims to None """
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)
