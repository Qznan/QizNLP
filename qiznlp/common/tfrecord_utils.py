#!/usr/bin/env python
# coding=utf-8
import os, re, glob
import numpy as np
import tensorflow as tf

""" tfrecord文件保存的名字后面均补上data数量
    e.g. xxx.tfrecord_1234
"""

TF_VERSION = int(tf.__version__.split('.')[1])


def tf_sparse_to_dense_new(v):
    if isinstance(v, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(v)
    return v


def tf_sparse_to_dense_old(v):
    if isinstance(v, tf.SparseTensor):
        return tf.sparse_to_dense(v.indices, v.dense_shape, v.values)
    return v


tf_sparse_to_densor = tf_sparse_to_dense_new if TF_VERSION >= 12 else tf_sparse_to_dense_old


def flat(l):
    # 平摊flatten
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def exist_tfrecord_file(tfrecord_file):
    files = glob.glob(f'{tfrecord_file}_*')
    files = list(filter(lambda f: re.match(r'^.*_\d+$', f), files))  # filter invalid
    return True if len(files) > 0 else False


def get_tfrecord_file(tfrecord_file):
    # 获得对应包括num的tfrecord名字
    files = glob.glob(f'{tfrecord_file}_*')
    files = list(filter(lambda f: re.match(r'^.*_\d+$', f), files))  # filter invalid
    if len(files) > 0:
        if len(files) > 1:
            files.sort(key=lambda f: os.path.getctime(f), reverse=True)  # 优先拿最新的
        return files[0]
    else:
        return None


def delete_exist_tfrecord_file(tfrecord_file):
    files = glob.glob(f'{tfrecord_file}_*')
    files = list(filter(lambda f: re.match(r'^.*_\d+$', f), files))  # filter invalid
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def add_num(file, num):
    return f'{file}_{num}'


def items2tfrecord(items, tfrecord_file):
    # 删除已有的
    delete_exist_tfrecord_file(tfrecord_file)

    def int_feat(value):
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def float_feat(value):
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def byte_feat(value):
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(bytesList=tf.train.BytesList(value=value))

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    count = 0
    for item in items:
        if count and not count % 100000:
            print(f'generating tfrecord... count: {count}')
        features = {}
        try:
            for k, v in item.items():
                # maybe_flatten
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    v = v.flatten()
                elif isinstance(v, list) and isinstance(v[0], list):
                    v = list(flat(v))

                ele = v[0] if isinstance(v, (list, np.ndarray)) else v  # 需检查元素的类型

                if isinstance(ele, (int, np.int, np.int32, np.int64)):
                    features[k] = int_feat(v)
                elif isinstance(ele, (float, np.float, np.float16, np.float32, np.float64)):
                    features[k] = float_feat(v)
                else:
                    features[k] = byte_feat(v)
            example = tf.train.Example(features=tf.train.Features(feature=features))
        except:
            print('error item:', item)
            continue
        writer.write(example.SerializeToString())
        count += 1
    writer.close()
    add_num_name = add_num(tfrecord_file, count)
    os.rename(tfrecord_file, add_num_name)
    if count == 0:
        raise Exception(f'error! count = {count} no example to save')
    print(f'save tfrecord file ok! {add_num_name} total count: {count}')
    return count


def tfrecord2dataset(tfrecord_files, feat_dct, shape_dct=None, batch_size=100, auto_pad=False, index=None, shard=None):
    """
    tf.VarLenFeature只能是平展后的1维数据，故原始数据是二维的话，如[None,4]，则需传入shape_dct进行reshape
    feat_dct = {
       'target': tf.FixedLenFeature([], tf.int64),
       's1': tf.VarLenFeature(tf.int64),
       's1_char': tf.VarLenFeature(tf.int64),
       'others': tf.FixedLenFeature([3], tf.string),
    }
    shape_dct = {
       's1_char': [-1, 4]  # if s1_char is 2-D Tensor and need to pad at the first dimension when batch
    }
    """
    # [注]该函数返回dataset的过程不需在图中,但后续迭代需在图中使用,如:
    # with self.graph.as_default():
    #     iterator = dataset.make_one_shot_iterator()
    #     dataset_features = iterator.get_next()
    # for i in train_step:
    #     features = sess.run(dataset_features)

    # 首先根据传入的tfrecord_file名字得到num
    if not isinstance(tfrecord_files, list):
        tfrecord_files = [tfrecord_files]
    total_count = 0
    files = []
    for file in tfrecord_files:
        if re.match(r'^.*_\d+$', file):  # 传入的是已经有数字了
            files.append(file)
            total_count += int(file.rsplit('_', 1)[1])
            continue
        file_ = get_tfrecord_file(file)
        if file_ is None:
            print(f'valid tfrecord(with num) file {file} is not found!')
            continue
        files.append(file_)
        total_count += int(file_.rsplit('_', 1)[1])
    tfrecord_files = files
    print(f'load tfrecord file ok! {" & ".join(tfrecord_files)} total count: {total_count}')

    def exm_parse(serialized_example):
        parsed_features = tf.parse_single_example(serialized_example, features=feat_dct)
        # VarLenFeature will return sparse tensor
        for k, v in parsed_features.items():
            parsed_features[k] = tf_sparse_to_densor(v)  # Convert sparse to dense when need
        # maybe need to reshape
        if shape_dct is not None:
            for name, shape in shape_dct.items():
                parsed_features[name] = tf.reshape(parsed_features[name], shape)
        return parsed_features

    def ints2int32(features):
        for k, v in features.items():
            if v.dtype in [tf.int64, tf.uint8]:
                features[k] = tf.cast(v, tf.int32)
        return features

    def padded_batch(dataset, batch_size, padded_shapes=None):
        if padded_shapes is None:
            padded_shapes = {k: [None] * len(shape) for k, shape in dataset.output_shapes.items()}
        return dataset.padded_batch(batch_size, padded_shapes)

    # tf.contrib.keras.preprocessing.sequence.pad_sequences

    dataset = tf.data.TFRecordDataset(tf.constant(tfrecord_files))
    # dataset = tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(tf.constant(data_files)))

    # 是否分布式训练数据分片, 上层传来的local_rank设置shard的数据，保证各个gpu采样的数据不重叠。
    if shard is not None and index is not None and index < shard:
        dataset = dataset.shard(shard, index)
        batch_size = int(np.ceil(batch_size / shard))  # batch再均分
        

    dataset = dataset.map(exm_parse)
    # dataset = dataset.map(exm_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # need tf >= 1.14

    dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)

    dataset = padded_batch(dataset, batch_size) if auto_pad else dataset.batch(batch_size)  # batch时是否要自动补齐

    dataset = dataset.map(ints2int32)  # 可向量化的操作放在batch后面以提高效率
    # dataset = dataset.map(ints2int32, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 可向量化的操作放在batch后面以提高效率

    dataset = dataset.repeat()  # repeat放在batch后面

    dataset = dataset.prefetch(2)  # pipline先异步准备好n个batch数据,训练step时数据同时处理
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # pipline先异步准备好n个batch数据,训练step时数据同时处理

    return dataset, total_count


def tfrecord2queue(tfrecord_files, feat_dct):
    # 返回后还需使用
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    # sess.run(tfrecord2queue)
    # coord.request_stop()
    # coord.join(threads)
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(tfrecord_files),
                                                    shuffle=None,
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feat_dct)

    for k in features:
        if features[k].dtype in [tf.int64, tf.uint8]:
            features[k] = tf.cast(features[k], tf.int32)

    # reshpae
    # features[k] = tf.reshape(features[k], [-1,1])
    sorted_keys = list(sorted(features))

    input_queue = tf.train.slice_input_producer([features[k] for k in sorted_keys], shuffle=False)
    # tf.data.Dataset.from_tensor_slices
    data = tf.train.batch(input_queue, batch_size=128, allow_smaller_final_batch=True, num_threads=8)
    ret = dict(zip(sorted_keys, data))
    return ret  # 返回的tensor供sess.run


def check_tfrecord(tfrecord_file, feat_dct):
    true_count = 0
    dataset, total_count = tfrecord2dataset(tfrecord_file, feat_dct)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    sess = tf.Session()
    while True:
        try:
            feat = sess.run(features)
            true_count += len(list(feat.values())[0])
        except Exception as e:
            print(e)
            print(true_count)
            raise e


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    #
    # feat_dct = {
    #     'contents': tf.FixedLenFeature([4, 50], tf.int64),
    #     'content_masks': tf.FixedLenFeature([4, 50], tf.int64),
    #     'content_lens': tf.FixedLenFeature([4], tf.int64),
    #     'char_contents': tf.FixedLenFeature([4, 50, 4], tf.int64),
    #     'char_content_masks': tf.FixedLenFeature([4, 50, 4], tf.int64),
    #     'char_content_lens': tf.FixedLenFeature([4, 50], tf.int64),
    #     'responses': tf.FixedLenFeature([50], tf.int64),
    #     'response_masks': tf.FixedLenFeature([50], tf.int64),
    #     'response_lens': tf.FixedLenFeature([], tf.int64),
    #     'char_responses': tf.FixedLenFeature([50, 4], tf.int64),
    #     'char_response_masks': tf.FixedLenFeature([50, 4], tf.int64),
    #     'char_response_lens': tf.FixedLenFeature([50], tf.int64),
    #     'targets': tf.FixedLenFeature([], tf.int64),
    #     'intents': tf.FixedLenFeature([4], tf.int64),
    # }
    # check_tfrecord('/dockerdata/yonaszhang/1223/data/match_ckpt_4_train_data.tfrecord_1', feat_dct)
    # exit(0)

    """ test """
    import time

    # items = [
    #     {'s1':[1,2,3,4],'s2':[5]},
    #     {'s1':[2,3,4],'s2':[6]},
    # ]
    # items2tfrecord(items, './test.tfrecord')
    feat_dct = {
        's1': tf.VarLenFeature(tf.int64),
        's2': tf.FixedLenFeature([], tf.int64),
    }
    dataset, _ = tfrecord2dataset('./test.tfrecord', feat_dct, batch_size=2, auto_pad=True)
    feature = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while True:
            print(sess.run(feature))
            time.sleep(1)
            # input('输入任意键以继续下一个')
