import os
import tensorflow as tf
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

from qiznlp.common.modules.common_layers import mask_nonpad_from_embedding
from qiznlp.common.modules.embedding import embedding
from qiznlp.common.modules.birnn import Bi_RNN
import qiznlp.common.utils as utils

conf = utils.dict2obj({
    'vocab_size': 1142,
    'embed_size': 300,
    'birnn_hidden_size': 300,
    'l2_reg': 0.0001,
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
            self.model_name = kwargs.get('model_name', 'esim')
            {
                'esim': self.build_model1,
                # add new here
            }[self.model_name]()
            print(f'model_name: {self.model_name} build graph ok!')

    def build_placeholder(self):
        # placeholder
        # 原则上模型输入输出不变，不需换新model
        self.s1 = tf.placeholder(tf.int32, [None, None], name='s1')
        self.s2 = tf.placeholder(tf.int32, [None, None], name='s2')
        self.target = tf.placeholder(tf.int32, [None], name="target")
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

    def build_model1(self):
        # embedding
        # [batch,len,embed]
        s1_embed, _ = embedding(tf.expand_dims(self.s1, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)
        s2_embed, _ = embedding(tf.expand_dims(self.s2, -1), conf.vocab_size, conf.embed_size, name='share_embedding', pretrain_embedding=conf.pretrain_emb)

        s1_input_mask = mask_nonpad_from_embedding(s1_embed)  # [batch,len1] 1 for nonpad; 0 for pad
        s2_input_mask = mask_nonpad_from_embedding(s2_embed)  # [batch,len2] 1 for nonpad; 0 for pad
        s1_seq_len = tf.cast(tf.reduce_sum(s1_input_mask, axis=-1), tf.int32)  # [batch]
        s2_seq_len = tf.cast(tf.reduce_sum(s2_input_mask, axis=-1), tf.int32)  # [batch]

        # bilstm sent encoder
        self.bilstm_encoder1 = Bi_RNN(cell_name='LSTMCell', hidden_size=conf.birnn_hidden_size, dropout_rate=self.dropout_rate)
        s1_bar, _ = self.bilstm_encoder1(s1_embed, s1_seq_len)  # [batch,len1,2hid]
        s2_bar, _ = self.bilstm_encoder1(s2_embed, s2_seq_len)  # [batch,len2,2hid]

        # local inference 局部推理
        with tf.variable_scope('local_inference'):
            # 点积注意力
            attention_logits = tf.matmul(s1_bar, tf.transpose(s2_bar, [0, 2, 1]))  # [batch,len1,len2]

            # 注意需attention mask  pad_mask * -inf + logits
            attention_s1 = tf.nn.softmax(attention_logits + tf.expand_dims((1. - s2_input_mask) * -1e9, 1))  # [batch,len1,len2]

            attention_s2 = tf.nn.softmax(tf.transpose(attention_logits, [0, 2, 1]) + tf.expand_dims((1. - s1_input_mask) * -1e9, 1))  # [batch,len2,len1]

            s1_hat = tf.matmul(attention_s1, s2_bar)  # [batch,len1,2hid]
            s2_hat = tf.matmul(attention_s2, s1_bar)  # [batch,len2,2hid]

            s1_diff = s1_bar - s1_hat
            s1_mul = s1_bar * s1_hat

            s2_diff = s2_bar - s2_hat
            s2_mul = s2_bar * s2_hat

            m_s1 = tf.concat([s1_bar, s1_hat, s1_diff, s1_mul], axis=2)  # [batch,len1,8hid]
            m_s2 = tf.concat([s2_bar, s2_hat, s2_diff, s2_mul], axis=2)  # [batch,len2,8hid]

        # composition 推理组成
        with tf.variable_scope('composition'):
            self.bilstm_encoder2 = Bi_RNN(cell_name='LSTMCell', hidden_size=conf.birnn_hidden_size, dropout_rate=self.dropout_rate)
            v_s1, _ = self.bilstm_encoder2(m_s1, s1_seq_len)  # [batch,len1,2hid]
            v_s2, _ = self.bilstm_encoder2(m_s2, s2_seq_len)  # [batch,len2,2hid]

            # average pooling  # 需将pad的vector变为0
            v_s1 = v_s1 * tf.expand_dims(s1_input_mask, -1)  # [batch,len1,2hid]
            v_s2 = v_s2 * tf.expand_dims(s2_input_mask, -1)  # [batch,len1,2hid]
            v_s1_avg = tf.reduce_sum(v_s1, axis=1) / tf.cast(tf.expand_dims(s1_seq_len, -1), tf.float32)  # [batch,2hid]
            v_s2_avg = tf.reduce_sum(v_s2, axis=1) / tf.cast(tf.expand_dims(s2_seq_len, -1), tf.float32)  # [batch,2hid]

            # max pooling  # 需将pad的vector变为极小值
            v_s1_max = tf.reduce_max(v_s1 + tf.expand_dims((1. - s1_input_mask) * -1e9, -1), axis=1)  # [batch,2hid]
            v_s2_max = tf.reduce_max(v_s2 + tf.expand_dims((1. - s2_input_mask) * -1e9, -1), axis=1)  # [batch,2hid]

            v = tf.concat([v_s1_avg, v_s1_max, v_s2_avg, v_s2_max], axis=-1)  # [batch,8hid]

        with tf.variable_scope('ffn'):
            h_ = tf.layers.dropout(v, rate=self.dropout_rate)
            h = tf.layers.dense(h_, 256, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0.0, 0.1))  # [batch,256]
            o_ = tf.layers.dropout(h, rate=self.dropout_rate)
            o = tf.layers.dense(o_, 1, kernel_initializer=tf.random_normal_initializer(0.0, 0.1))  # [batch,1]
        self.logits = tf.squeeze(o, -1)  # [batch]

        # loss
        with tf.name_scope('loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.target, tf.float32), logits=self.logits)  # [batch]
            loss = tf.reduce_mean(loss, -1)  # scalar

            l2_reg = conf.l2_reg
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_reg
            loss += l2_loss

        self.loss = loss
        self.y_prob = tf.nn.sigmoid(self.logits)
        self.y_prob = tf.identity(self.y_prob, name='y_prob')

        with tf.name_scope("accuracy"):
            self.correct = tf.equal(
                tf.cast(tf.greater_equal(self.y_prob, 0.5), tf.int32),
                self.target)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

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
            token_ids = token_ids[:max_word_len]
        # token_ids = utils.pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
        return token_ids  # [len]

    def create_feed_dict_from_data(self, data, ids, mode='train'):
        # data:数据已经转为id, data不同字段保存该段字段全量数据
        batch_s1 = [data['s1'][i] for i in ids]
        batch_s2 = [data['s2'][i] for i in ids]
        if len(set([len(e) for e in batch_s1])) != 1:  # 长度不等
            batch_s1 = utils.pad_sequences(batch_s1, padding='post')
        if len(set([len(e) for e in batch_s2])) != 1:  # 长度不等
            batch_s2 = utils.pad_sequences(batch_s2, padding='post')
        feed_dict = {
            self.s1: batch_s1,
            self.s2: batch_s2,
            self.target: [data['target'][i] for i in ids],
        }
        if mode == 'train': feed_dict['num'] = len(batch_s1)
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_features(self, features, mode='train'):
        # feature:tfrecord数据的example, 每个features的不同字段包括该字段一个batch数据
        feed_dict = {
            self.s1: features['s1'],
            self.s2: features['s2'],
            self.target: features['target'],
        }
        if mode == 'train': feed_dict['num'] = len(features['s1'])
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.
        return feed_dict

    def create_feed_dict_from_raw(self, batch_s1, batch_s2, batch_y, token2id_dct, mode='infer'):
        word2id = token2id_dct['word2id']
        
        feed_s1 = [self.sent2ids(s1, word2id) for s1 in batch_s1]
        feed_s2 = [self.sent2ids(s2, word2id) for s2 in batch_s2]

        feed_dict = {
            self.s1: utils.pad_sequences(feed_s1, padding='post'),
            self.s2: utils.pad_sequences(feed_s2, padding='post'),
        }
        feed_dict[self.dropout_rate] = conf.dropout_rate if mode == 'train' else 0.

        if mode == 'infer':
            return feed_dict

        if mode in ['train', 'dev']:
            assert batch_y, 'batch_y should not be None when mode is train or dev'
            feed_dict[self.target] = batch_y
            return feed_dict
        
        raise ValueError(f'mode type {mode} not support')

    @classmethod
    def generate_data(cls, file, token2id_dct):
        word2id = token2id_dct['word2id']
        data = {
            's1': [],
            's2': [],
            'target': []
        }
        with open(file, 'r', encoding='U8') as f:
            for i, line in enumerate(f):
                item = line.strip().split('\t')
                if len(item) != 3:
                    print('error', repr(line))
                    continue
                s1 = item[0]
                s2 = item[1]
                y = item[2]
                s1_ids = cls.sent2ids(s1, word2id, max_word_len=50)
                s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                y_id = int(y)
                if i < 5:  # check
                    print(f'check {i}:')
                    print(f'{s1} -> {s1_ids}')
                    print(f'{s2} -> {s2_ids}')
                    print(f'{y} -> {y_id}')
                data['s1'].append(s1_ids)
                data['s2'].append(s2_ids)
                data['target'].append(y_id)
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
                    if len(item) != 3:
                        print('error', repr(line))
                        continue
                    try:
                        s1 = item[0]
                        s2 = item[1]
                        y = item[2]
                        s1_ids = cls.sent2ids(s1, word2id, max_word_len=50)
                        s2_ids = cls.sent2ids(s2, word2id, max_word_len=50)
                        y_id = int(y)
                        if i < 5:  # check
                            print(f'check {i}:')
                            print(f'{s1} -> {s1_ids}')
                            print(f'{s2} -> {s2_ids}')
                            print(f'{y} -> {y_id}')
                        d = {
                            's1': s1_ids,
                            's2': s2_ids,
                            'target': y_id,
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
            'target': tf.FixedLenFeature([], tf.int64),
        }
        dataset, count = tfrecord2dataset(tfrecord_file, feat_dct, batch_size=batch_size, auto_pad=True, index=index, shard=shard)
        return dataset, count

    def get_signature_export_model(self):
        inputs_dct = {
            's1': self.s1,
            's2': self.s2,
            'dropout_rate': self.dropout_rate,
        }
        outputs_dct = {
            'y_prob': self.y_prob,
        }
        return inputs_dct, outputs_dct

    @classmethod
    def get_signature_load_pbmodel(cls):
        inputs_lst = ['s1', 's2', 'dropout_rate']
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
        model.s1 = graph.get_tensor_by_name('s1:0')
        model.s2 = graph.get_tensor_by_name('s2:0')
        model.dropout_rate = graph.get_tensor_by_name('dropout_rate:0')
        # self.target = self.graph.get_tensor_by_name('target:0')
        model.y_prob = graph.get_tensor_by_name('y_prob:0')

        saver.restore(sess, ckpt_name)
        print(f':: restore success! {ckpt_name}')
        return model, saver
