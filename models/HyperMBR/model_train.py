#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
import os
from utils import recommendationResult
def train(model, epoch, loader, optim, args, loss_func):
    log_interval = args.log_freq
    model.train()
    start = time()
    for i, data in enumerate(loader):
        users_b, bundles = data
        # torch.cuda.empty_cache()
        modelout = model(users_b.to(args.device), bundles.to(args.device),epoch)
        loss = loss_func(modelout, batch_size=loader.batch_size)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # lr_scheduler.step()
        if i % log_interval == 0:
            print('U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                       100. * (i+1) / len(loader), loss))

    print('epoch:{},item_Teach_bundle_KL_loss:{},bundle_Teach_item_KL_loss:{},kd_loss:{}'.format(epoch,model.item_Teach_bundle_KL_loss,model.bundle_Teach_item_KL_loss,model.kdLossSum))
    model.item_Teach_bundle_KL_loss=0
    model.bundle_Teach_item_KL_loss=0
    model.kdLossSum=0
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss



def test(model, loader, args, metrics,metrics_itemLevel,metrics_bundleLevel,epoch):
    # if epoch % args.recommendation_result_interval==0:
    #     rr = recommendationResult.RecommendationResult(os.path.join(
    #         str(args.recommendation_result_interval), args.dataset, args.model)
    #         ,epoch)
    model.eval()
    for metric in metrics:
        metric.start()
    for metric in metrics_itemLevel:
        metric.start()
    for metric in metrics_bundleLevel:
        metric.start()

    start = time()
    with torch.no_grad():
        rs = model.propagate()
        for users, ground_truth_u_b, train_mask_u_b in loader:
            item_level_scores,bundle_level_scores,scores = model.evaluate(rs, users.to(args.device))
            train_mask_u_b=train_mask_u_b.to(args.device)

            item_level_scores -= 1e8*train_mask_u_b
            bundle_level_scores-= 1e8*train_mask_u_b
            scores-= 1e8*train_mask_u_b

            ground_truth_u_b=ground_truth_u_b.to(args.device)
            for metric in metrics:
                metric(scores,ground_truth_u_b )
            for metric in metrics_itemLevel:
                metric(item_level_scores, ground_truth_u_b)
            for metric in metrics_bundleLevel:
                metric(bundle_level_scores, ground_truth_u_b)

            # if epoch % args.recommendation_result_interval == 0:
            #     rr.saveRecommendationResult(users, metrics)
    print('time={:d}s'.format(int(time()-start)))
    print('epoch:{}'.format(epoch+1))

    print('item Level + Bundle Level:')
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('\n')
    print('item Level:')
    for metric in metrics_itemLevel:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('\n')
    print('Bundle Level:')
    for metric in metrics_bundleLevel:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('\n')

    print('itemLevel Encoder curvatures:')
    for layer in model.itemLevel_encoder.layers:
        print('c_in',layer.c_in,'c_out:',layer.c_out)
    print('bundleLevel Encoder curvatures:')
    for layer in model.bundleLevel_encoder.layers:
        print('c_in',layer.c_in,'c_out:',layer.c_out)

    return metrics

