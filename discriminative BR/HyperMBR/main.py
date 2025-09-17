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
    metrics_val_itemLevel = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]
    metrics_val_bundleLevel = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]
    metrics_test = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]
    metrics_test_itemLevel = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]
    metrics_test_bundleLevel = [Recall(5), NDCG(5), Recall(10), NDCG(10), Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80),
                    NDCG(80)]


    TARGET = 'Recall@20'

    # log
    val_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                            checkpoint_target=TARGET, task='tune')
    val_itemLevel_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='val_itemLevel')
    val_bundleLevel_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='val_bundleLevel')
    test_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='test')
    test_itemLevel_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='test_itemLevel')
    test_bundleLevel_log = logger.Logger(os.path.join(args.log, args.dataset, f"{args.model}_{args.manifold}"), 'best',
                             checkpoint_target=TARGET, task='test_bundleLevel')

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    # if not args.lr_reduce_freq:
    #     args.lr_reduce_freq = args.epochs

    itemLevelCoefs=[0.7]
    # itemLevelCoefs=[0.7]
    # itemLevel_cs=[1.0,2.0,3.0,4.0,5.0]
    # bundleLevel_cs = [1.0, 2.0, 3.0, 4.0, 5.0]
    itemLevel_cs=[1.0]
    bundleLevel_cs = [2.0]
    temperatures=[1.0]
    for lr, dropout, dim, negative_num, num_layers, manifold,itemLevelCoef,itemLevel_c,bundleLevel_c,temperature in product(args.lr, args.dropout, args.dim,
                                                                        args.negative_num, args.num_layers,
                                                                        args.manifold,itemLevelCoefs,itemLevel_cs,bundleLevel_cs,temperatures):
        print("lr:{}, dropout:{}, dim:{}, negative_num:{}, num_layers:{}, manifold:{},itemLevel_C:{},bundleLevel_C:{},temperature:{}".format(lr, dropout, dim,
                                                                                              negative_num, num_layers,
                                                                                              manifold,itemLevel_cs,bundleLevel_c,temperature))
        args.itemLevelCoef=itemLevelCoef
        args.bundleLevelCoef=1-itemLevelCoef
        args.itemLevel_c=itemLevel_c
        args.bundleLevel_c=bundleLevel_c
        args.temperature=temperature
        # loadDataSet
        bundle_train_data, bundle_val_data, bundle_test_data, item_data, assist_data = get_dataset(args, negative_num)

        train_loader = DataLoader(bundle_train_data, 4096, True,
                                  num_workers=16, pin_memory=True)
        val_loader = DataLoader(bundle_val_data, 2048, False,
                                num_workers=16, pin_memory=True)
        test_loader = DataLoader(bundle_test_data, 2048, False,
                                 num_workers=16, pin_memory=True)
        # train_loader = DataLoader(bundle_train_data, 16384, True,
        #                           num_workers=16, pin_memory=True)
        # val_loader = DataLoader(bundle_val_data, 1024, False,
        #                         num_workers=16, pin_memory=True)
        # test_loader = DataLoader(bundle_test_data, 1024, False,
        #                          num_workers=16, pin_memory=True)
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

        model = BRECModel_Distance(args, modelInfo_tune, graph)

        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=lr,
                                                        weight_decay=args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=int(args.lr_reduce_freq),
        #     gamma=float(args.gamma)
        # )
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")

        env = {
            'lr': args.lr,
            'op': args.optimizer,  # Adam
            'dataset': args.dataset,
            'model': args.model,
            'itemLevel_c': args.itemLevel_c,
            'bundleLevel_c': args.bundleLevel_c,
            'use_distillation':args.use_distillation,
            'warmup':args.warmup,
            'itemLevelCoef':args.itemLevelCoef,
            'bundleLevelCoef':args.bundleLevelCoef,
            'temperature': args.temperature,
        }
        print('env:',env)
        if args.cuda is not None and int(args.cuda) >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
            model = model.to(args.device)
            # for x, val in data.items():
            #     if torch.is_tensor(data[x]):
            #         data[x] = data[x].to(args.device)

            #  continue training
        if args.sample == 'hard' and args.conti_train != None:
            model.load_state_dict(torch.load(args.conti_train))
            print('load model and continue training')
        # Train model
        # t_total = time.time()
        # counter = 0
        # best_val_metrics = model.init_metric_dict()
        # best_test_metrics = None
        # best_emb = None

        retry = 0
        while retry >= 0:
            val_log.update_modelinfo(modelInfo_tune, env, metrics_val)
            val_itemLevel_log.update_modelinfo(modelInfo_tune, env, metrics_val_itemLevel)
            val_bundleLevel_log.update_modelinfo(modelInfo_tune, env, metrics_val_bundleLevel)

            test_log.update_modelinfo(modelInfo_test, env, metrics_test)
            test_itemLevel_log.update_modelinfo(modelInfo_test, env, metrics_test_itemLevel)
            test_bundleLevel_log.update_modelinfo(modelInfo_test, env, metrics_test_bundleLevel)

            early = args.early
            for epoch in range(args.epochs):
                trainloss = train(model, epoch + 1, train_loader, optimizer, args, loss_func)

                if epoch % args.eval_freq == 0:
                    print('Val:')
                    output_metrics1 = test(model, val_loader, args, metrics_val,metrics_val_itemLevel,metrics_val_bundleLevel, epoch)
                    print('Test:')
                    output_metrics2 = test(model, test_loader, args, metrics_test,metrics_test_itemLevel,metrics_test_bundleLevel, epoch)
                    # logging.info()
                    val_log.update_log(metrics_val, model)
                    val_itemLevel_log.update_log(metrics_val_itemLevel,model)
                    val_bundleLevel_log.update_log(metrics_val_bundleLevel,model)

                    test_log.update_log(metrics_test, model)
                    test_itemLevel_log.update_log(metrics_test_itemLevel,model)
                    test_bundleLevel_log.update_log(metrics_test_bundleLevel,model)

                    # check overfitting
                    if epoch > 10:
                        if check_overfitting(val_log.metrics_log, TARGET, 1, show=False):
                            print('########check_overfitting#########')
                            break
                    # early stop
                    early = early_stop(val_log.metrics_log[TARGET], early, threshold=0)
                    print("early:", early)
                    if early <= 0:
                        print('########early_stop#########')
                        break

            val_log.close_log(TARGET)
            val_itemLevel_log.close_log(TARGET)
            val_bundleLevel_log.close_log(TARGET)
            test_log.close_log(TARGET)
            test_itemLevel_log.close_log(TARGET)
            test_bundleLevel_log.close_log(TARGET)
            print("Optimization Finished!")
            retry -= 1

    val_log.close()
    val_itemLevel_log.close()
    val_bundleLevel_log.close()
    test_log.close()
    test_itemLevel_log.close()
    test_bundleLevel_log.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
