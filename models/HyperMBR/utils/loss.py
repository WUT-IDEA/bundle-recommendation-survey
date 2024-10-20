#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F
class _Loss(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. 
        `none`: no reduction will be applied, 
        `mean`: the sum of the output will be divided by the number of elements in the output, 
        `sum`: the output will be summed. 

        Note: size_average and reduce are in the process of being deprecated, 
        and in the meantime,  specifying either of those two args will override reduction. 
        Default: `sum`
        '''
        super().__init__()
        assert(reduction == 'mean' or reduction ==
               'sum' or reduction == 'none')
        self.reduction = reduction

class BPRLoss(_Loss):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        # ensure reduction in (mean，sum, none)
        super().__init__(reduction)

    def forward(self, model_output, **kwargs):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive bundles/items, column 1 must be the negative.
        '''
        pred, L2_loss = model_output
        # BPR loss
        loss = -torch.log(torch.sigmoid(pred[:, 0] - pred[:, 1]))
        # reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        loss += L2_loss / kwargs['batch_size'] if 'batch_size' in kwargs else 0
        return loss




# def kl_loss(student_logits,teacher_logits):
#     log_pred_student = F.log_softmax(student_logits,dim=1)
#     pred_teacher = F.softmax(teacher_logits,dim=1)
#     # loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
#     kl_loss=F.kl_div(log_pred_student, pred_teacher, size_average=False) / student_logits.shape[1]
#     return kl_loss
# student向teacher靠近







#WMRBLoss
class WMRBLoss(_Loss):
    def __init__(self, args,reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        # ensure reduction in (mean，sum, none)
        super().__init__(reduction)
        self.args=args
        self.use_distillation=args.use_distillation


    def forward(self, model_output, **kwargs):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive bundles/items, column 1 must be the negative.
        '''
        # itemLevel_distance,bundleLevel_distance, L2_loss, CL_loss  = model_output
        pred_rank, kd_loss= model_output

        # item_To_bundle_KL_loss= kl_loss(itemLevel_prob, bundleLevel_prob)
        # bundle_To_item_DKD_loss=dkd_loss(itemLevel_prob,bundleLevel_prob)
        # item_To_bundle_DKD_loss = dkd_loss(bundleLevel_prob, itemLevel_prob)


        # itemLevel_distance+bundleLevel_distance
        # itemLevel_postive_distance=itemLevel_distance[:,0].reshape(-1,1).repeat(1,itemLevel_distance.shape[1]-1)
        # itemLevel_negative_distance=itemLevel_distance[:,1:].reshape(itemLevel_distance.shape[0],itemLevel_distance.shape[1]-1)
        #
        # itemLevel_pred_rank=torch.max(itemLevel_postive_distance-itemLevel_negative_distance+torch.ones_like(itemLevel_postive_distance)*2,torch.zeros_like(itemLevel_postive_distance))
        # itemLevel_pred_rank=torch.sum(itemLevel_pred_rank,dim=1).reshape(1,-1)
        #
        # # bundleLevel_distance+bundleLevel_distance
        # bundleLevel_postive_distance=bundleLevel_distance[:,0].reshape(-1,1).repeat(1,bundleLevel_distance.shape[1]-1)
        # bundleLevel_negative_distance=bundleLevel_distance[:,1:].reshape(bundleLevel_distance.shape[0],bundleLevel_distance.shape[1]-1)
        #
        # bundleLevel_pred_rank=torch.max(bundleLevel_postive_distance-bundleLevel_negative_distance+torch.ones_like(bundleLevel_postive_distance)*2,torch.zeros_like(bundleLevel_postive_distance))
        # bundleLevel_pred_rank=torch.sum(bundleLevel_pred_rank,dim=1).reshape(1,-1)
        # dkd_loss_val=kl_loss(itemLevel_pred_rank,bundleLevel_pred_rank)/ kwargs['batch_size'] if 'batch_size' in kwargs else 0

        # pred_rank=itemLevel_pred_rank+bundleLevel_pred_rank

        # WMRB loss
        loss = torch.log(1 + pred_rank)

        # loss+=min(kwargs["epoch"] / self.warmup, 1.0) *item_To_bundle_DKD_loss

        # loss=pred_rank
        # loss+=0.4*CL_loss
        # loss +=  0.5*(bundle_To_item_DKD_loss+item_To_bundle_DKD_loss)
        # loss+=item_To_bundle_DKD_loss+bundle_To_item_KL_loss
        # loss+=dkd_loss_val
        # reduction
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be  'none' | 'mean' | 'sum'")
        if self.use_distillation:
            loss+=kd_loss
        # loss += L2_loss / kwargs['batch_size'] if 'batch_size' in kwargs else 0
        return loss