"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from manifolds.hyperboloid import Hyperboloid


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    # tocoo转化后，graph包含data，row，col
    values = graph.data
    # vstack,将矩阵按照列的方向进行叠加，hstack按照行的方向进行叠加
    # indices就变成了2*n的矩阵，上下两行分别是row和col
    indices = np.vstack((graph.row, graph.col))
    # 这些转化都是为了符合FloatTensor的参数要求
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
                                     torch.Size(graph.shape))
    return graph


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, info):
        super(BaseModel, self).__init__()
        self.info = info
        self.args = args
        self.manifold_name = info.manifold
        # # 固定曲度，默认是1.0
        # if args.c is not None:
        #     self.c = torch.tensor([args.c])
        #     if not args.cuda == -1:
        #         self.c = self.c.to(args.device)
        # # 可训练的曲度
        # else:
        #     self.c = nn.Parameter(torch.Tensor([3.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = info.embedding_size + 1
        self.num_users = args.num_users
        self.num_bundles = args.num_bundles
        self.num_items = args.num_items
        self.embed_L2_norm = args.embed_L2_norm


    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            # 如果是hyperboloid的话，在每个输入最前面加个0
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class Info(object):
    '''
    [FOR `utils.logger`]

    the base class that packing all hyperparameters and infos used in the related model
    '''

    def __init__(self, embedding_size, embed_L2_norm):
        assert isinstance(embedding_size, int) and embedding_size > 0
        self.embedding_size = embedding_size
        assert embed_L2_norm >= 0
        self.embed_L2_norm = embed_L2_norm

    def get_title(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(lambda x: dct[x].get_title() if isinstance(dct[x], Info) else x, dct.keys()))

    def get_csv_title(self):
        return self.get_title().replace('\t', ', ')

    def __getitem__(self, key):
        if hasattr(self, '_info'):
            return self._info[key]
        else:
            return self.__getattribute__(key)

    def __str__(self):
        dct = self.__dict__
        if '_info' in dct:
            # pop,字典删除特定元素
            dct.pop('_info')
        # map(str,dct.values()) ,讲dct中所有的值转化为字符串，str相当于一个function
        return '\t'.join(map(str, dct.values()))

    def get_line(self):
        return self.__str__()

    def get_csv_line(self):
        return self.get_line().replace('\t', ', ')


class BRECModel_Distance_Info(Info):
    def __init__(self, embedding_size, dropout, num_layers, manifold, negativeNum, act, embed_L2_norm, task='tune'):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > dropout >= 0
        self.dropout = dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers
        assert isinstance(negativeNum, int) and negativeNum > 0
        self.negativeNum = negativeNum
        assert manifold in ['Euclidean', 'Hyperboloid', 'PoincareBall']
        self.manifold = manifold
        self.task = task


class BRECModel_Distance(BaseModel):
    """
    Base model for bundle recommendation task.
    """

    def __init__(self, args, info, graph):
        super(BRECModel_Distance, self).__init__(args, info)

        assert isinstance(graph, list)
        self.graph = graph
        self.epison = 1e-8
        self.margin = 2
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.a=nn.Parameter(torch.Tensor([0.3]),requires_grad=True)
        self.warmup=args.warmup
        self.itemLevelCoef=args.itemLevelCoef
        self.bundleLevelCoef=args.bundleLevelCoef
        self.item_Teach_bundle_KL_loss = 0
        self.bundle_Teach_item_KL_loss = 0
        self.kdLossSum=0
        self.temperature =args.temperature
        # self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        ub_graph, ui_graph, bi_graph = graph
        #  deal with weights
        # multiply对位相乘 ，sqrt（矩阵） 矩阵每个元素对自己开根号
        # ravel --拉成一维数组
        # 1/(sqrt(bundle所属的item数量))
        # 相当于每一行的数据都除以了 每一行1的数量
        bi_norm = sp.diags(1 / (np.sqrt((bi_graph.multiply(bi_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ bi_graph

        # bb_graph(a,b) 表示bundle a 和bundle b之间的重叠度,或者叫相似度
        bb_graph = bi_norm @ bi_norm.T

        if ui_graph.shape == (self.num_users, self.num_items):
            # add self-loop
            # bmat就是build matrix
            # sp.identity 单位矩阵
            # 这里在前面和后面加上单位矩阵，是为了模拟直接在原矩阵主对角线上加1的操作，效果是一样的
            # itemLevel_graph -- user-item融合之后的关系图，为了方便进行图卷积
            itemLevel_graph = sp.bmat([[sp.identity(ui_graph.shape[0]), ui_graph],
                                       [ui_graph.T, sp.identity(ui_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")

        self.itemLevel_graph = to_tensor(laplace_transform(itemLevel_graph)).to(args.device)
        print('finish generating itemLevel graph')

        if ub_graph.shape == (self.num_users, self.num_bundles) \
                and bb_graph.shape == (self.num_bundles, self.num_bundles):
            # add self-loop
            # 这个地方可能存在优化的可能，就是， 左上角的那个user-user 的矩阵可以不简单的初始化为单位矩阵，可以是用户之间的关系矩阵
            # bundleLevel_graph -- user-bundle-bundle之间的关系图
            bundleLevel_graph = sp.bmat([[sp.identity(ub_graph.shape[0]), ub_graph],
                                         [ub_graph.T, sp.identity(ub_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")

        self.bundleLevel_graph = to_tensor(laplace_transform(bundleLevel_graph)).to(args.device)
        print('finish generating non-itemLevel graph')

        # pooling_graph是为了itempropagate 之后对bundle所属的item的特征进行整合用的
        #  pooling graph
        bundle_size = bi_graph.sum(axis=1) + self.epison
        # bi_graph每一行也都除了 bundle所含item的数量
        bi_graph = sp.diags(1 / bundle_size.A.ravel()) @ bi_graph
        self.pooling_graph = to_tensor(bi_graph).to(args.device)

        if args.create_embeddings:
            self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, args.feat_dim))
            # 初始化，符合正态分布
            nn.init.xavier_normal_(self.users_feature)
            self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, args.feat_dim))
            nn.init.xavier_normal_(self.bundles_feature)
            self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, args.feat_dim))
            nn.init.xavier_normal_(self.items_feature)

        if args.itemLevel_c is None:
            self.itemLevel_c=torch.nn.Parameter(torch.Tensor([3.0])).to(args.device)
        else:
            self.itemLevel_c=torch.Tensor([args.itemLevel_c]).to(args.device)

        if args.bundleLevel_c is None:
            self.bundleLevel_c=torch.nn.Parameter(torch.Tensor([3.0])).to(args.device)
        else:
            self.bundleLevel_c=torch.Tensor([args.bundleLevel_c]).to(args.device)

        self.itemLevel_encoder = getattr(encoders, args.model)(self.itemLevel_c, info, args,levelflag='itemLevel')
        self.bundleLevel_encoder = getattr(encoders, args.model)(self.bundleLevel_c, info, args,levelflag='bundleLevel')

        self.gate=torch.nn.Linear(args.feat_dim*3,2)


    def CL_distance_loss(self,embA,embB,c):
        pos = embA[:, 0, :]
        aug = embB[:, 0, :]

        # pos = F.normalize(pos, p=2, dim=1)
        # aug = F.normalize(aug, p=2, dim=1)

        pos_score =  -self.manifold.sqdist(pos, aug,c)#torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score= -self.manifold.sqdist(pos.unsqueeze(dim=1), aug, c).squeeze(dim=-1)
        # ttl_score=torch.full_like(ttl_score,50)-ttl_score
        # ttl_socre=50-ttl_score
        # ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def sum_CL_loss(self,users_feature,bundles_feature,c):
        users_feature_itemLevel, users_feature_bundleLevel = users_feature  # batch_n_f
        bundles_feature_itemLevel, bundles_feature_bundleLevel = bundles_feature  # batch_n_f

        user_cross_loss=self.CL_distance_loss(users_feature_itemLevel, users_feature_bundleLevel,c)
        bundle_cross_loss=self.CL_distance_loss(bundles_feature_itemLevel, bundles_feature_bundleLevel,c)
        CL_losses = [user_cross_loss, bundle_cross_loss]

        CL_loss = sum(CL_losses) / len(CL_losses)
        return CL_loss

    def dkd_loss(self,logits_student, logits_teacher, alpha=1.0, beta=8.0, temperature=2):
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        # cat_mask就是将softmax的结果，根据gt_mask,other_mask，全部加在一起，形成1*2的矩阵，[all_target_probability_sum,all_non_target_probability_sum]
        pred_student = torch.cat(
            (pred_student[:, 0].reshape(-1, 1), torch.sum(pred_student[:, 1:], dim=1, keepdim=True)), dim=1)
        pred_teacher = torch.cat(
            (pred_teacher[:, 0].reshape(-1, 1), torch.sum(pred_teacher[:, 1:], dim=1, keepdim=True)), dim=1)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
                * (temperature ** 2)
        )

        logits_teacher_part2 = logits_teacher[:, 1:]
        logits_student_part2 = logits_student[:, 1:]

        pred_teacher_part2 = F.softmax(
            logits_teacher_part2 / temperature, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student_part2 / temperature, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
                * (temperature ** 2)
        )
        return alpha * tckd_loss + beta * nckd_loss

    def cal_dkd_loss(self,itemLevel_distance,bundleLevel_distance,epoch):
        itemLevel_prob = self.dc(itemLevel_distance)
        bundleLevel_prob = self.dc(bundleLevel_distance)
        # item_Teach_bundle_DKD_loss = self.dkd_loss(bundleLevel_prob, itemLevel_prob)
        bundle_Teach_item_DKD_loss = self.dkd_loss(itemLevel_prob, bundleLevel_prob)

        dkd_loss_sum = 0.5 * item_Teach_bundle_DKD_loss + 0.5 * bundle_Teach_item_DKD_loss
        #dkd_loss_sum =  bundle_Teach_item_DKD_loss
        dkd_loss = min(epoch / self.warmup, 1.0) * dkd_loss_sum
        return dkd_loss

    def kl_loss(self,student_logits, teacher_logits,temperature=2.0):
        pred_log_student = F.log_softmax(student_logits / temperature, dim=1)
        pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(pred_log_student, pred_teacher, reduction='batchmean')
        return kl_loss

    def cal_kd_loss(self,itemLevel_distance,bundleLevel_distance,epoch):
        # itemLevel_prob=self.dc(itemLevel_distance)
        # bundleLevel_prob=self.dc(bundleLevel_distance)

        itemLevel_prob=itemLevel_distance
        bundleLevel_prob=bundleLevel_distance

        # item_and_bundle_prob=self.dc(itemLevel_distance+bundleLevel_distance)
        item_Teach_bundle_KL_loss = self.kl_loss(bundleLevel_prob, itemLevel_prob,self.temperature)
        bundle_Teach_item_KL_loss =  self.kl_loss(itemLevel_prob, bundleLevel_prob,self.temperature)

        kd_loss_sum=self.bundleLevelCoef*bundle_Teach_item_KL_loss+self.itemLevelCoef*item_Teach_bundle_KL_loss
        self.item_Teach_bundle_KL_loss+=item_Teach_bundle_KL_loss
        self.bundle_Teach_item_KL_loss+=bundle_Teach_item_KL_loss
        #kd_loss_sum=bundle_Teach_item_KL_loss

        kd_loss = min(epoch / self.warmup, 1.0) * kd_loss_sum
        # print('epoch:{},item_Teach_bundle_KL_loss:{},bundle_Teach_item_KL_loss:{},kd_loss:{}'.format(epoch,item_Teach_bundle_KL_loss,bundle_Teach_item_KL_loss,kd_loss))
        self.kdLossSum+=kd_loss
        return kd_loss

    def forward(self, users, bundles,epoch):
        users_feature, bundles_feature = self.propagate()
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        pred_rank,kd_loss = self.getPredRankAndKDLoss(users_embedding, bundles_embedding,epoch)

        # CL_loss=self.sum_CL_loss(users_embedding,bundles_embedding,self.itemLevel_encoder.layers[-1].c_out)
        # L2_loss = self.regularize(users_embedding, bundles_embedding)

        return pred_rank,kd_loss#, L2_loss#,CL_loss

    def propagate(self):
        #  ============================= item level propagation  =============================
        itemLevel_features = torch.cat((self.users_feature, self.items_feature), 0)

        # 调用编码器--HyperGCN对输入的特征进行重新编码，此时的输出还是处于双曲空间
        itemLevel_encode_features = self.itemLevel_encoder.encode(itemLevel_features, self.itemLevel_graph)
        # itemLevel_encode_features=F.normalize(itemLevel_encode_features)
        itemLevel_users_feature, itemLevel_items_feature = torch.split(itemLevel_encode_features, (
        self.users_feature.shape[0], self.items_feature.shape[0]), 0)

        c_out = self.itemLevel_encoder.layers[-1].c_out
        itemLevel_items_feature_tangent = self.manifold.logmap0(itemLevel_items_feature, c=c_out)
        itemLevel_bundles_feature = F.normalize(torch.matmul(self.pooling_graph, itemLevel_items_feature_tangent))
        # itemLevel_bundles_feature = torch.matmul(self.pooling_graph, itemLevel_items_feature_tangent)
        itemLevel_bundles_feature = self.manifold.proj(self.manifold.expmap0(itemLevel_bundles_feature, c=c_out),
                                                       c=c_out)

        #  ============================= bundle level propagation =============================
        bundleLevel_features = torch.cat((self.users_feature, self.bundles_feature), 0)
        bundleLevel_encode_features = self.bundleLevel_encoder.encode(bundleLevel_features, self.bundleLevel_graph)
        # bundleLevel_encode_features=F.normalize(bundleLevel_encode_features)
        bundleLevel_users_feature, bundleLevel_bundles_feature = torch.split(bundleLevel_encode_features, (
        self.users_feature.shape[0], self.bundles_feature.shape[0]), 0)

        users_feature = [itemLevel_users_feature, bundleLevel_users_feature]
        bundles_feature = [itemLevel_bundles_feature, bundleLevel_bundles_feature]

        return users_feature, bundles_feature

    def regularize(self, users_feature, bundles_feature):
        users_feature_itemLevel, users_feature_bundleLevel = users_feature  # batch_n_f
        bundles_feature_itemLevel, bundles_feature_bundleLevel = bundles_feature  # batch_n_f

        L2_loss = self.embed_L2_norm * \
                  ((users_feature_itemLevel ** 2).sum() + (users_feature_bundleLevel ** 2).sum() + \
                   (bundles_feature_itemLevel ** 2).sum() + (bundles_feature_bundleLevel ** 2).sum())
        return L2_loss

    def getPredRankAndKDLoss(self, users_embedding, bundles_embedding,epoch):
        users_feature_itemLevel, users_feature_bundleLevel = users_embedding
        bundles_feature_itemLevel, bundles_feature_bundleLevel = bundles_embedding
        itemLevel_cOut = self.itemLevel_encoder.layers[-1].c_out
        bundleLevel_cOut = self.bundleLevel_encoder.layers[-1].c_out
        if self.manifold.name == 'Hyperboloid':
            itemLevel_distance = self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel,
                                                      itemLevel_cOut).squeeze(dim=2)
            bundleLevel_distance = self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel,
                                                        bundleLevel_cOut).squeeze(dim=2)
        else:
            itemLevel_distance = self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel,
                                                      itemLevel_cOut)
            bundleLevel_distance = self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel,
                                                        bundleLevel_cOut)

        distance=itemLevel_distance+bundleLevel_distance

        kd_loss=self.cal_kd_loss(itemLevel_distance,bundleLevel_distance,epoch)
        # kd_loss=self.cal_dkd_loss(itemLevel_distance,bundleLevel_distance,epoch)

        postive_distance=distance[:,0].reshape(-1,1).repeat(1,distance.shape[1]-1)
        negative_distance=distance[:,1:].reshape(distance.shape[0],distance.shape[1]-1)

        pred_rank=torch.max(postive_distance-negative_distance+torch.ones_like(postive_distance)*2,torch.zeros_like(postive_distance))
        pred_rank=torch.sum(pred_rank,dim=1)
        return pred_rank,kd_loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all bundles for `users` by `propagate_result`
        '''
        users_feature, bundles_feature = propagate_result
        users_feature_itemLevel, users_feature_bundleLevel = [i[users] for i in users_feature]  # batch_f
        bundles_feature_itemLevel, bundles_feature_bundleLevel = bundles_feature  # b_f

        itemLevel_cOut = self.itemLevel_encoder.layers[-1].c_out
        bundleLevel_cOut = self.bundleLevel_encoder.layers[-1].c_out
        users_feature_itemLevel = torch.unsqueeze(users_feature_itemLevel, dim=1)
        users_feature_bundleLevel = torch.unsqueeze(users_feature_bundleLevel, dim=1)
        if self.manifold.name == 'Hyperboloid':
            bundle_level_scores = torch.squeeze(
                -(self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel, bundleLevel_cOut)),
                dim=2)
            item_level_scores = torch.squeeze(
                -(self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel, itemLevel_cOut)),
                dim=2)
            scores=bundle_level_scores+item_level_scores
        else:
            scores = -(self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel, itemLevel_cOut) + \
                       self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel, bundleLevel_cOut))
        # if self.manifold.name == 'Hyperboloid':
        #     scores = torch.squeeze(
        #         -(self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel, itemLevel_cOut) + \
        #           self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel, bundleLevel_cOut)),
        #         dim=2)
        # else:
        #     scores = -(self.manifold.sqdist(users_feature_itemLevel, bundles_feature_itemLevel, itemLevel_cOut) + \
        #                self.manifold.sqdist(users_feature_bundleLevel, bundles_feature_bundleLevel, bundleLevel_cOut))

        return item_level_scores,bundle_level_scores,scores
