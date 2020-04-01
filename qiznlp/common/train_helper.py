#!/usr/bin/env python
# coding=utf-8
import os, pickle
import numpy as np
from .tfrecord_utils import exist_tfrecord_file
from . import utils


def prepare_tfrecord(raw_data_file,
                     model,
                     token2id_dct,
                     tokenize,
                     preprocess_raw_data_fn,
                     save_data_prefix,
                     save_data_dir='../data',
                     update_txt=False,
                     update_tfrecord=False,
                     **kwargs):
    # 1. read raw_data_file
    # 2. preprocess (seg/split...) -> .train .dev .test
    # 3. generate tfrecord
    # 4. load tfrecord
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    # define data file name
    train_txt_file = f'{save_data_dir}/{save_data_prefix}_train_data.txt'
    dev_txt_file = f'{save_data_dir}/{save_data_prefix}_dev_data.txt'
    test_txt_file = f'{save_data_dir}/{save_data_prefix}_test_data.txt'
    train_tfrecord_file = f'{save_data_dir}/{save_data_prefix}_train_data.tfrecord'
    dev_tfrecord_file = f'{save_data_dir}/{save_data_prefix}_dev_data.tfrecord'
    test_tfrecord_file = f'{save_data_dir}/{save_data_prefix}_test_data.tfrecord'

    if update_txt:
        update_tfrecord = True
        if os.path.exists(train_txt_file): os.remove(train_txt_file)
        if os.path.exists(dev_txt_file): os.remove(dev_txt_file)
        if os.path.exists(test_txt_file): os.remove(test_txt_file)

    if not all([exist_tfrecord_file(train_tfrecord_file), exist_tfrecord_file(dev_tfrecord_file)]) or update_tfrecord:  # 没有或需要更新tfrecord
        # 首先检查txt file
        if not all([os.path.exists(train_txt_file), os.path.exists(dev_txt_file)]):
            train_data, dev_data, test_data = preprocess_raw_data_fn(raw_data_file, tokenize=tokenize, token2id_dct=token2id_dct, **kwargs)
            if train_data:
                utils.list2file(train_txt_file, train_data)
                print(f'generate train txt file ok! {train_txt_file}')
            if dev_data:
                utils.list2file(dev_txt_file, dev_data)
                print(f'generate dev txt file ok! {dev_txt_file}')
            if test_data:
                utils.list2file(test_txt_file, test_data)
                print(f'generate test txt file ok! {test_txt_file}')
        
        if os.path.exists(train_txt_file):
            model.generate_tfrecord(train_txt_file, token2id_dct, train_tfrecord_file)
        if os.path.exists(dev_txt_file):
            model.generate_tfrecord(dev_txt_file, token2id_dct, dev_tfrecord_file)
        if os.path.exists(test_txt_file):
            model.generate_tfrecord(test_txt_file, token2id_dct, test_tfrecord_file)

    return train_tfrecord_file, dev_tfrecord_file, test_tfrecord_file


def prepare_pkldata(raw_data_file,
                    model,
                    token2id_dct,
                    tokenize,
                    preprocess_raw_data_fn,
                    save_data_prefix,
                    save_data_dir='../data',
                    update_txt=False,
                    update_pkl=False,
                    **kwargs):
    # 返回data: 各个字段有全量数据
    # 同时返回训练及验证的数据数量
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    # define data file name
    train_txt_file = f'{save_data_dir}/{save_data_prefix}_train_data.txt'
    dev_txt_file = f'{save_data_dir}/{save_data_prefix}_dev_data.txt'
    test_txt_file = f'{save_data_dir}/{save_data_prefix}_test_data.txt'
    train_pkl_file = f'{save_data_dir}/{save_data_prefix}_train_data.pkl'
    dev_pkl_file = f'{save_data_dir}/{save_data_prefix}_dev_data.pkl'
    test_pkl_file = f'{save_data_dir}/{save_data_prefix}_test_data.pkl'

    if update_txt:
        update_pkl = True
        if os.path.exists(train_txt_file): os.remove(train_txt_file)
        if os.path.exists(dev_txt_file): os.remove(dev_txt_file)
        if os.path.exists(test_txt_file): os.remove(test_txt_file)

    if not all([os.path.exists(train_pkl_file), os.path.exists(dev_pkl_file)]) or update_pkl:  # 没有或需要更新pkldata
        # 首先检查txt file
        if not all([os.path.exists(train_txt_file), os.path.exists(dev_txt_file)]):
            train_data, dev_data, test_data = preprocess_raw_data_fn(raw_data_file, tokenize=tokenize, token2id_dct=token2id_dct, **kwargs)
            if train_data:
                utils.list2file(train_txt_file, train_data)
                print(f'generate train txt file ok! {train_txt_file}')
            if dev_data:
                utils.list2file(dev_txt_file, dev_data)
                print(f'generate dev txt file ok! {dev_txt_file}')
            if test_data:
                utils.list2file(test_txt_file, test_data)
                print(f'generate test txt file ok! {test_txt_file}')

        if os.path.exists(train_txt_file):
            train_pkldata = model.generate_data(train_txt_file, token2id_dct)
            pickle.dump(train_pkldata, open(train_pkl_file, 'wb'))
            print(f'generate and save train pkl file ok! {train_pkl_file}')
        if os.path.exists(dev_txt_file):
            dev_pkldata = model.generate_data(dev_txt_file, token2id_dct)
            pickle.dump(dev_pkldata, open(dev_pkl_file, 'wb'))
            print(f'generate and save train pkl file ok! {dev_pkl_file}')
        if os.path.exists(test_txt_file):
            test_pkldata = model.generate_data(test_txt_file, token2id_dct)
            pickle.dump(test_pkldata, open(test_pkl_file, 'wb'))
            print(f'generate and save train pkl file ok! {test_pkl_file}')
            
    return train_pkl_file, dev_pkl_file, test_pkl_file


# 得到当前图中所有变量的名称
# tensor_names = [tensor.name for tensor in graph.as_graph_def().node]  # 得到当前图中所有变量的名称
# for tensor_name in tensor_names:
#     if not tensor_name.startswith('save') and not tensor_name.startswith('gradients'):
#         if 'Adam' not in tensor_name and 'Initializer' not in tensor_name:
#             print(tensor_name)


def gen_pos_neg_sample(items, sample_idx, num_neg_exm=9, seed=1234):
    import random
    import copy
    total_exm = []
    random.seed(seed)  # 采样随机负样本的随机性
    cands_ids = list(range(len(items)))

    for i, exm in enumerate(items):
        exm.append(1)  # add pos label
        total_exm.append(exm)  # add pos exm

        neg_exm_lst = []
        neg_sample_set = set()

        while len(neg_exm_lst) < min(num_neg_exm, len(items) - 1):  # data数量比负采样样本数还少的情况
            neg_id = random.choice(cands_ids)
            if neg_id == i:  # 如果采到自己
                continue
            if items[neg_id][sample_idx] == exm[sample_idx]:  # 如果sample的文本相同
                continue
            if items[neg_id][sample_idx] in neg_sample_set:  # 如果sample已经采样了
                continue
            neg_sample_set.add(items[neg_id][sample_idx])
            neg_exm = copy.deepcopy(exm)
            neg_exm[sample_idx] = items[neg_id][sample_idx]  # change to neg 
            neg_exm[-1] = 0  # add neg label
            neg_exm_lst.append(neg_exm)  # add neg exm
          
        total_exm.extend(neg_exm_lst)  # add neg exms
        
    return total_exm


def calc_recall(epo_s1, epo_prob, epo_y, topk=5, strip_pad=False):
    comb = list(zip(epo_s1, epo_prob, epo_y))
    s1_dct = {}
    for s1, prob, y in comb:
        if 'ndarray' in str(type(s1)):
            s1 = s1.flatten().tolist()
        if strip_pad:
            while s1[-1] == 0:
                s1.pop()
        s1 = tuple(s1)
        if s1 not in s1_dct:
            s1_dct[s1] = []
        s1_dct[s1].append([prob, y])
    for s1 in s1_dct:
        s1_dct[s1].sort(key=lambda e: e[0], reverse=True)
    total_recall = []
    for s1 in s1_dct:
        y_lst = [e[1] for e in s1_dct[s1]]
        # print(len(y_lst))
        if len(y_lst) < 2:  # 没有正负样本
            continue
        # print(sum(y_lst))
        if sum(y_lst) != 1:  # 真实标签中没有正样本1 或超过1个正样本
            continue
        recall_item = [sum(y_lst[:topk]) for topk in range(1, topk+1)]
        total_recall.append(recall_item)
    total_recall = np.mean(total_recall, axis=0).tolist()
    return total_recall