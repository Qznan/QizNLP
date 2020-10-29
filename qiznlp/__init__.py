#!/usr/bin/env python
# encoding=utf-8

import os
import shutil

def qiznlp_init():
    cwd = os.getcwd()
    curr_dir = os.path.dirname(__file__)

    print('copying ...')

    shutil.copytree(f'{curr_dir}/model', f'{cwd}/model')
    print('copy model_dir finish')

    shutil.copytree(f'{curr_dir}/run', f'{cwd}/run')
    os.remove(f'{cwd}/run/run_base.py')
    print('copy run_dir finish')

    shutil.copytree(f'{curr_dir}/deploy', f'{cwd}/deploy')
    print('copy deploy_dir finish')

    shutil.copytree(f'{curr_dir}/data', f'{cwd}/data')
    print('copy data_dir finish')

    shutil.copytree(f'{curr_dir}/common/modules/bert/chinese_L-12_H-768_A-12',
                    f'{cwd}/common/modules/bert/chinese_L-12_H-768_A-12')
    print('copy bert_model_dir finish')



