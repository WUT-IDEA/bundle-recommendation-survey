#!/usr/bin/env python3
# -*- coding: utf-8 -*-
CONFIG = {
    'name': '@4879',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'MIDGN',
    'dataset_name': 'Youshu',
    'task': 'tune',
    'eval_task': 'test',

    ## optimal hyperparameters
    'lrs': [1e-2],
    'message_dropouts': [0.3],
    'node_dropouts': [0],
    'decays': [1e-7],

    ## hard negative sample and further train
    'sample': 'simple',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.3, 0.3], # probability 0.8
    'conti_train': 'log/Youshu/',

    ## other settings
    'epochs': 1000,
    'early': 50,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test':['log/Youshu']
}

