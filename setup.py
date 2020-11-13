#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='QizNLP',
    version='0.1.4',
    author='Qznan',
    author_email='summerzynqz@gmail.com',
    description='Quick run NLP in many task',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MPLv2.0',
    url='https://github.com/Qznan/QizNLP',
    packages=find_packages(),
    package_data={'qiznlp':
                      ['data/*.txt',
                       'common/modules/bert/chinese_L-12_H-768_A-12/*.json',
                       'common/modules/bert/chinese_L-12_H-768_A-12/*.txt',
                       ],
                  },
    install_requires=[
        'jieba',
        'tensorflow>=1.8, <=1.14'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'qiznlp_init=qiznlp:qiznlp_init',
        ]
    },
    keywords='NLP Classification Match Sequence_Label Senquence_to_Senquence Neural_Network',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Environment :: MacOS X',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
