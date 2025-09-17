#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import BRECModel_Distance, BRECModel_Distance_Info
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, check_overfitting, early_stop
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.eval_utils import Recall, NDCG
from utils.loss import BPRLoss, WMRBLoss
from model_train import train, test
from utils import logger
from itertools import product
from utils import recommendationResult


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu')
    args.patience = args.epochs if not args.patience else int(args.patience)

    # metric
    metrics_val = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                   NDCG(80)]
    metrics_test = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]
    TARGET = 'Recall@5'

    # log
    val_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                            checkpoint_target=TARGET, task='tune')
    test_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='test')
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    # if not args.lr_reduce_freq:
    #     args.lr_reduce_freq = args.epochs
    for lr, dropout, dim, negative_num, num_layers, manifold in product(args.lr, args.dropout, args.dim,
                                                                        args.negative_num, args.num_layers,
                                                                        args.manifold):
        print("lr:{}, dropout:{}, dim:{}, negative_num:{}, num_layers:{}, manifold:{}".format(lr, dropout, dim,
                                                                                              negative_num, num_layers,
                                                                                              manifold))
        # loadDataSet
        bundle_train_data, bundle_val_data, bundle_test_data, item_data, assist_data = get_dataset(args, negative_num)

        train_loader = DataLoader(bundle_train_data, 4096, True,
                                  num_workers=16, pin_memory=True)
        val_loader = DataLoader(bundle_val_data, 2048, False,
                                num_workers=16, pin_memory=True)
        test_loader = DataLoader(bundle_test_data, 2048, False,
                                 num_workers=16, pin_memory=True)
        #  graph
        ub_graph = bundle_train_data.ground_truth_u_b
        ui_graph = item_data.ground_truth_u_i
        bi_graph = assist_data.ground_truth_b_i
        graph = [ub_graph, ui_graph, bi_graph]

        args.num_users = bundle_train_data.num_users
        args.num_bundles = bundle_train_data.num_bundles
        args.num_items = bundle_train_data.num_items

        if args.use_distillation:
            print("use distillation,warmup:{},dkd_alpha:{},dkd_beta:{},dkd_temperature:{}".format(args.warmup,
                                                                                                  args.dkd_alpha,
                                                                                                  args.dkd_beta,
                                                                                                  args.dkd_temperature))
        # loss
        # loss_func = BPRLoss('mean')
        loss_func = WMRBLoss(args, 'mean')
        # Model and optimizer
        modelInfo_tune = BRECModel_Distance_Info(embedding_size=dim, dropout=dropout, num_layers=num_layers,
                                                 manifold=manifold, negativeNum=negative_num, act=args.act,
                                                 embed_L2_norm=args.embed_L2_norm, task='tune')
        modelInfo_test = BRECModel_Distance_Info(embedding_size=dim, dropout=dropout, num_layers=num_layers,
                                                 manifold=manifold, negativeNum=negative_num, act=args.act,
                                                 embed_L2_norm=args.embed_L2_norm, task='test')

        model = BRECModel_Distance(args, modelInfo_test, graph).to(args.device)
        model.load_state_dict(torch.load('./1_eac9d5_Recall@5.pth'), False)
        # test model, loader, args, metrics,epoch
        test(model, test_loader, args , metrics_test,1)


    val_log.close()
    test_log.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)