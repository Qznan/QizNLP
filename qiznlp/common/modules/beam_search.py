#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from tensorflow.python.util import nest

INF = 1e7
# INF = float('inf')

# debug
TF_PRINT = False


def tf_print(tensor, message=''):
    if not TF_PRINT:
        return tensor
    else:
        return tf.Print(tensor, [tf.shape(tensor), tensor], message='Debug:%s' % message, summarize=40)


def log_prob_from_logits(logits, axis=-1):
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=True)


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


def assign_value_to_tensor(tensor, idx, value):
    """ tensor 1-D
        idx scalar
        value scalar
        ret 1-D
    """
    dtype = tensor.dtype
    length = tf.shape(tensor)[0]

    weight_one_target_idx = tf.one_hot(idx, length, on_value=1., off_value=0.)
    weight_zero_target_idx = 1. - weight_one_target_idx

    weight_one_target_idx = tf.cast(weight_one_target_idx, dtype=dtype)
    weight_zero_target_idx = tf.catt(weight_zero_target_idx, dtype=dtype)
    ret = tensor * weight_zero_target_idx + value * weight_one_target_idx
    return ret


def multinomial_without_repeat(logits, num_samples, top_k, uniform=False, eos_id=1):
    """ logits 2-D
        num_samples scalar
        topk scalar
        uniform bool
        ret [batch,num_samples]
    """
    batch_size = tf.shape(logits)[0]
    init_i = tf.constant(0)
    init_logits = logits
    init_ret = tf.zeros([batch_size, 0], dytpe=tf.int32)

    # 先把eos设置为不可能(-INF)
    init_logits = tf.map_fn(lambda x: assign_value_to_tensor(x, eos_id, value=-INF), init_logits)

    init_topk_logits, init_topk_ids = tf.nn.top_k(init_logits, k=top_k, sorted=False)  # [batch,topk]
    if uniform:
        init_topk_logits = tf.zeros_like(init_topk_logits)

    def loop_condition(i, *_):
        # i < N and i < topk
        return tf.logical_and(tf.less(i, num_samples), tf.less(i, top_k))

    def loop_body(i, logits, ret):
        sample_idx = tf.multinomial(logits, 1, output_dtype=tf.int32)  # [batch,1]
        # 计算原始vocab_vector里面的idx
        batch_pos = tf.reshape(tf.range(batch_size), [batch_size, 1])  # [batch,1]
        ids_pos = tf.stack([batch_pos, sample_idx], axis=-1)  # [batch,1,2]
        ori_sample_idx = tf.gather_nd(init_topk_ids, ids_pos)  # [batch,1]

        ret = tf.concat([ret, ori_sample_idx], axis=-1)
        sample_idx = tf.squeenze(sample_idx, axis=-1)  # [batch]
        logits = tf.map_fn(lambda x: (assign_value_to_tensor(x[0], x[1], value=-INF), x[1]), (logits, sample_idx))  # tuple(logits,tmp)
        logits = logits[0]
        i += 1
        return i, logits, ret

    final_i, final_logits, final_ret = tf.while_loop(loop_condition,
                                                     loop_body,
                                                     [init_i, init_topk_logits, init_ret],
                                                     shape_invariants=[tf.TensorShape([]),
                                                                       tf.TensorShape([None, None]),
                                                                       tf.TensorShape([None, None])]
                                                     )
    return final_ret


def merge_pre_n_dim(tensor, n=2):
    """ n=2 [a,b,...] -> [a*b,...] """
    shape = shape_list(tensor)
    for i in range(1, n):
        shape[0] *= shape[i]
    shape = shape[:1] + shape[n:]
    return tf.reshape(tensor, shape)


def unmerge_first_dim(tensor, sizes):
    """ sizes=[a,b] [a*b,...] -> [a,b,...] """
    shape = shape_list(tensor)
    shape = sizes + shape[1:]
    return tf.reshape(tensor, shape)


def expand_and_tile_to_ndims(tensor, sizes):
    """ sizes=[beam] [a,...] -> [a,beam,...] """
    n_dims = len(sizes)
    for _ in range(n_dims):
        tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    for i in range(n_dims):
        tile_dims[i + 1] = sizes[i]
    return tf.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
    """ return the shape of tensor but set middle dims to None """
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def compute_batch_and_group_indices(batch_size, group_size, beam_size):
    batch_pos = tf.range(batch_size * group_size * beam_size) // (group_size * beam_size)
    batch_pos = tf.reshape(batch_pos, [batch_size, group_size, beam_size])

    group_pos = tf.range(group_size * beam_size) // beam_size
    group_pos = tf.reshape(group_pos, [group_size, beam_size])  # [group,beam]
    group_pos = tf.expand_dims(group_pos, axis=0)
    group_pos = tf.tile(group_pos, [batch_size, 1, 1])  # [batch,group,beam]

    return tf.stack([batch_pos, group_pos], axis=3)  # [batch,group,beam,2


def beam_search(symbols_to_logits_fn,
                initial_ids,  # int32
                beam_size,
                decode_length,
                vocab_size,
                alpha=0,
                states=None,
                eos_id=1,
                stop_early=True,
                num_group=1,
                gamma=0,
                stop_nums=None,
                top_k=30):
    """ """
    group_size = num_group  # 换个名字
    beam_size = beam_size // group_size
    use_group = tf.not_equal(group_size, 1)  # 是否分组flag

    batch_size = shape_list(initial_ids)[0]
    # initial log probs
    init_log_probs = tf.concat([tf.constant([0.]), tf.tile(tf.constant([-INF]), [beam_size - 1])], axis=0)  # [beam]
    init_log_probs = tf.expand_dims(tf.expand_dims(init_log_probs, axis=0), axis=0)  # [1,1,beam]
    init_log_probs = tf.tile(init_log_probs, [batch_size, group_size, 1])  # [batch,group,beam]

    # initial seq
    init_seq = expand_and_tile_to_ndims(initial_ids, [group_size, beam_size])  # [batch,group,beam]
    init_seq = tf.expand_dims(init_seq, axis=3)  # [batch,group,beam,1] 使用pad当做eos

    # initial state
    if states:  # [batch,...]
        states = nest.map_structure(lambda t: expand_and_tile_to_ndims(t, [group_size, beam_size]), states)  # [batch,group,beam,...]

    finished_flags = tf.fill([batch_size, group_size, beam_size], False)  # [batch,group,beam]

    def loop_body(i, seq, log_probs, finished_flags, states):
        """ """
        # flat the seq
        flat_ids = tf.reshape(seq, [batch_size * group_size * beam_size, -1])  # [batch*group*beam,length]
        if states:
            flat_states = nest.map_structure(lambda t: merge_pre_n_dim(t, n=3), states)
            flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i, flat_states)
            states = nest.map_structure(lambda t: unmerge_first_dim(t, [batch_size, group_size, beam_size]), flat_states)
        else:
            flat_logits = symbols_to_logits_fn(flat_ids, i, states)

        logits = tf.reshape(flat_logits, [batch_size, group_size, beam_size, -1])  # [batch,group,beam,vocab]

        # calc log-prob
        candidate_log_probs = log_prob_from_logits(logits)

        def process_diversity():
            """ 多样性鼓励因子，1个group中的多样性 """
            _candidate_log_probs = tf.reshape(candidate_log_probs, [-1, vocab_size])
            _, topk_candidate_log_probs_ids = tf.nn.top_k(_candidate_log_probs, k=vocab_size)
            diversity_rank = tf.map_fn(tf.invert_permutation, topk_candidate_log_probs_ids) + 1
            diversity_rank = tf.to_float(diversity_rank) * -1. * gamma
            _candidate_log_probs += diversity_rank
            _candidate_log_probs = tf.reshape(_candidate_log_probs, [batch_size, group_size, beam_size, vocab_size])
            return _candidate_log_probs

        # 如果部分组则应从第一步开始加上多样性因子，如果分组应从第二步开始
        start_step = tf.cond(use_group, lambda: 1, lambda: 0)
        candidate_log_probs = tf.cond(tf.greater(i, start_step), process_diversity, lambda: candidate_log_probs)

        # 使结束的句子下一步强制选择eos, 通过调整下一步的candidate_log_probs
        eos_one_hot_vec = tf.one_hot(eos_id, vocab_size, on_value=1., off_value=0.)  # [vocab]
        vocab_mask = -1 * eos_one_hot_vec + INF * (1 - eos_one_hot_vec)  # [vocab]
        vocab_mask = tf.expand_dims(vocab_mask, axis=0)  # [1,vocab]
        flat_finished_flags = tf.reshape(tf.to_float(finished_flags), [-1, 1])  # [batch,group,beam,1]

        mask = tf.matmul(flat_finished_flags, vocab_mask)  # [batch,group,beam,vocab]
        mask += 1
        mask = tf.reshape(mask, [batch_size, group_size, beam_size, vocab_size])

        candidate_log_probs *= mask

        # 加上前面已计算分数的log_probs
        log_probs = candidate_log_probs + tf.expand_dims(log_probs, axis=3)  # [batch,group,beam,vocab]

        # 平展beam和vocab用以组内全局比较
        flat_log_probs = tf.reshape(log_probs, [-1, group_size, beam_size * vocab_size])  # [batch,group,beam*vocab]

        def group_first_step_sample_from_logits():
            """ 根据logits进行首字采样，必须分组才能使用本功能"""
            batch_logits = tf.reshape(flat_logits, [batch_size, group_size, beam_size, -1])  # [batch,group,beam,vocab]
            batch_logits = batch_logits[:, 0, 0, :]  # [batch,vocab]  # 由于首字每个beam里的所有logits一样
            sample_idx = multinomial_without_repeat(batch_logits, group_size, top_k=top_k, uniform=False)  # [batch,group]
            # idx 变换为[batch*group*beam*1] 利于下一步取出对应log_prob
            sample_idx = tf.expand_dims(sample_idx, axis=-1)  # [batch,group,1]
            sample_idx = tf.tile(sample_idx, [1, 1, beam_size])  # [batch,group,beam]
            sample_idx = tf.reshape(sample_idx, [-1, 1])  # [batch*group*beam,1]
            # 取出对应log_prob分数
            flat_log_prob = log_prob_from_logits(flat_logits)  # [batch*group*beam,vocab]
            bgbid = tf.expand_dims(tf.range(batch_size * group_size * beam_size), axis=-1)  # [batch*group*beam,1]
            bgbid = tf.concat([bgbid, sample_idx], axis=-1)  # [batch*group*beam,2]
            sample_log_prob = tf.gather_nd(flat_log_prob, bgbid)  # [batch*group*beam,vocab]
            # 变换为正式形式
            sample_idx = tf.reshape(sample_idx, [batch_size, group_size, beam_size])  # [batch,group,beam]
            sample_log_prob = tf.reshape(sample_log_prob, [batch_size, group_size, beam_size])  # [batch,group,beam]
            # 每组mask除第一个以外的分数，适配分组beam_search的第二步
            log_prob_mask = tf.ones_like(sample_log_prob) * -INF
            sample_log_prob = tf.concat([sample_log_prob[:, :, :1], log_prob_mask[:, :, 1:]], axis=-1)
            sample_log_prob = tf_print(sample_log_prob, 'sample_log_prob')
            sample_idx = tf_print(sample_idx, 'sample_idx')
            return sample_log_prob, sample_idx

        def group_first_step_argmax_from_logprob():
            """ 需每组的开头第一个字是相同的，组间不同
                e.g.
                batch=2 group=3 beam=2
                ret
                ids [[a,a],[b,b],[c,c]]
                scores [[a,-inf],[b,-inf],[c,-inf]]
            """
            _topk_log_probs, _topk_ids = tf.nn.top_k(flat_log_probs, k=group_size)  # [batch,group,group]
            _topk_ids = _topk_ids[:, 0, :]  # [batch,group]
            _topk_log_probs = _topk_log_probs[:, 0, :]  # [batch,group]

            _topk_ids = tf.expand_dims(_topk_ids, axis=-1)  # [batch,group,1]
            _topk_ids = tf.tile(_topk_ids, [1, 1, beam_size])  # [batch,group,beam]

            _topk_log_probs = tf.expand_dims(_topk_log_probs, axis=-1)  # [batch,group,1]
            # 每组mask掉除第一个以外的分数,适配分组beam_search的第二步
            log_prob_mask = tf.ones([batch_size, group_size, beam_size]) * -INF  # [batch,group,beam]
            _topk_log_probs = tf.concat([_topk_log_probs, log_prob_mask[:, :, 1:]], axis=2)

            return _topk_log_probs, _topk_ids

        def other_step():
            _topk_log_probs, _topk_ids = tf.nn.top_k(flat_log_probs, k=beam_size)  # [batch,group,beam]
            return _topk_log_probs, _topk_ids

        # 如果不分组，就不执行first_step了
        use_group_first_step = tf.cond(use_group, lambda: 0, lambda: -1)
        topk_log_probs, topk_ids = tf.cond(tf.equal(i, use_group_first_step), group_first_step_argmax_from_logprob, other_step)

        next_log_probs = topk_log_probs  # [batch,group,beam]

        # calc what group the top probs are in
        topk_seq_idx = topk_ids // vocab_size  # [batch,group,beam]
        topk_ids %= vocab_size  # unflatten the ids [batch,group,beam]

        batch_and_group_pos = compute_batch_and_group_indices(batch_size, group_size, beam_size)  # [batch,group,beam,2]
        topk_coord = tf.concat([batch_and_group_pos, tf.expand_dims(topk_seq_idx, axis=-1)], axis=3)  # [batch,group,beam,3]

        finished_flag = tf.equal(topk_ids, eos_id)
        if states:
            states = nest.map_structure(lambda s: tf.gather_nd(s, topk_coord), states)

        next_seq = tf.gather_nd(seq, topk_coord)
        next_seq = tf.concat([next_seq, tf.expand_dims(topk_ids, axis=3)], axis=3)

        return i + 1, next_seq, next_log_probs, finished_flag, states

    def loop_condition(i, unused_seq, unused_log_probs, finished_flags, unused_states):
        not_all_finish = tf.logical_not(tf.reduce_all(finished_flags))
        less_than_maxlength = tf.less(i, decode_length)
        return tf.logical_and(not_all_finish, less_than_maxlength)

    # start loop
    _, seq, log_probs, finished_flags, _ = tf.while_loop(
        loop_condition,
        loop_body,
        [tf.constant(0), init_seq, init_log_probs, finished_flags, states],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, None]),
            # finished_flags.get_shape(),
            nest.map_structure(get_state_shape_invariants, states)]
    )

    # 合并group和beam
    seq = tf.reshape(seq, [batch_size, group_size * beam_size, -1])  # [batch,beam*group,len]
    log_probs = tf.reshape(log_probs, [batch_size, group_size * beam_size])  # [batch,beam*group]

    seq = tf_print(seq, 'seq')
    log_probs = tf_print(log_probs, 'log_probs')

    return seq[:, :, 1:], log_probs  # 去除起始符


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


def greedy_search(symbols_to_logits_fn,
                  initial_ids,  # int32
                  max_decode_len,
                  cache=None,
                  eos_id=1,
                  sampling_method='argmax',
                  sampling_temp=0.):
    """  """
    # initial_ids: [batch] tf.zeros([batch_size], dtype=tf.int32) 
    batch_size = shape_list(initial_ids)[0]
    decoded_ids = tf.expand_dims(initial_ids, 1)  # [batch,1]
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    hit_eos = tf.fill([batch_size], False)

    def inner_loop(i, hit_eos, decoded_ids, cache, log_prob):
        """ one step for greedy decoding """
        logits, cache = symbols_to_logits_fn(decoded_ids, i, cache)
        log_probs = log_prob_from_logits(logits)
        temperature = 0.0 if sampling_method == 'argmax' else sampling_temp
        next_id = tf.cast(sample_with_temperature(logits, temperature), dtype=tf.int32)  # [batch]
        hit_eos |= tf.equal(next_id, eos_id)

        log_prob_indices = tf.stack([tf.range(tf.to_int32(batch_size)), next_id], axis=1)
        log_prob += tf.gather_nd(log_probs, log_prob_indices)

        decoded_ids = tf.concat([decoded_ids, tf.expand_dims(next_id, axis=1)], axis=1)
        return i + 1, hit_eos, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
        finished = (i >= max_decode_len)
        finished |= tf.reduce_all(hit_eos)
        return tf.logical_not(finished)

    _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop,
        [tf.constant(0), hit_eos, decoded_ids, cache, initial_log_prob],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            nest.map_structure(get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])

    decoded_ids = tf.expand_dims(decoded_ids, 1)[:, :, 1:]  # 去除齐师傅
    log_prob = tf.expand_dims(log_prob, 1)
    return decoded_ids, log_prob  # [batch,1,len], [batch,1]


def symbols_to_logits_fn_readme():
    """ 
        输入：ids, i, cache
            ids: [batch,len] 一般直接取最末尾一个进行计算
        输出：logits, cache
            logits: [batch,h]  len维度没有
    """
    pass
