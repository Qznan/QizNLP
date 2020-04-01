#!/usr/bin/env python
# encoding=utf-8

import os
import shutil

def qiznlp_init():
    cwd = os.getcwd()
    curr_dir = os.path.dirname(__file__)

    print('copying ...')
    shutil.copytree(curr_dir + '/model', cwd + '/model')
    print('copy model-dir finish')
    shutil.copytree(curr_dir + '/run', cwd + '/run')
    os.remove(cwd + '/run/run_base.py')
    print('copy run-dir finish')
    shutil.copytree(curr_dir + '/data', cwd + '/data')
    print('copy data-dir finish')
    # shutil.copytree(curr_dir + '/common', cwd + '/common')
    # print('copy common-dir finish')

