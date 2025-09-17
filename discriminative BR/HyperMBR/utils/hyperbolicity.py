import os
import pickle as pkl
import sys
import time

import networkx
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
import scipy.sparse as sp

# from utils.data_utils import load_data_lp


def hyperbolicity_sample(G, num_samples=1000):
    curr_time = time.time()
    hyps = []
    node_tuple_cache=[]
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        node_tuple=tuple(sorted(list(node_tuple)))

        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            temp=(s[-1] - s[-2]) / 2
            # if temp==2.0:
            #     print(node_tuple)
            hyps.append(temp)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    hyps_statistic = {}
    for i in range(len(hyps)):
        if hyps[i] in hyps_statistic.keys():
            hyps_statistic[hyps[i]] += 1
        else:
            hyps_statistic[hyps[i]] = 1
    print(hyps_statistic)
    return max(hyps), np.mean(hyps)

def loadDataset(path, name, fileName):
    with open(os.path.join(path, name, fileName), 'r') as f:
        U_B_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indice = np.array(U_B_pairs)
        values = np.ones(len(U_B_pairs))
        return indice, values


def getHyperbolicityOfDataset(path='../data/', DatasetName='Youshu', fileName='user_bundle_train.txt'):
    with open(os.path.join(path, DatasetName, '{}_data_size.txt'.format(DatasetName)), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    U_B_I_graph = sp.dok_matrix((num_users + num_items + num_bundles, num_users + num_items + num_bundles))
    print(num_users , num_items , num_bundles)
    indice, values = loadDataset(path, DatasetName, fileName)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    U_B_graph = coo_matrix((values, (rowIndice, colIndice)), shape=(num_users, num_bundles)).tocsr().astype(np.int8)

    fileName = 'user_item.txt'
    U_I_indice, u_i_values = loadDataset(path, DatasetName, fileName)
    U_I_rowIndice = U_I_indice[:, 0]
    U_I_colIndice = U_I_indice[:, 1]
    U_I_graph = csr_matrix((u_i_values, (U_I_rowIndice, U_I_colIndice)), shape=(num_users, num_items)).tocsr().astype(np.int8)

    fileName = 'bundle_item.txt'
    B_I_indice, b_i_values = loadDataset(path, DatasetName, fileName)
    B_I_rowIndice = B_I_indice[:, 0]
    B_I_colIndice = B_I_indice[:, 1]
    # b_i_values = np.zeros_like(b_i_values)
    B_I_graph = csr_matrix((b_i_values, (B_I_rowIndice, B_I_colIndice)), shape=(num_bundles, num_items)).tocsr().astype(np.int8)

    u_u=csr_matrix(np.zeros((U_B_graph.shape[0],U_B_graph.shape[0]),dtype=np.int8))
    b_b=csr_matrix(np.zeros((U_B_graph.shape[1],U_B_graph.shape[1]),dtype=np.int8))
    i_i=csr_matrix(np.zeros((U_I_graph.shape[1],U_I_graph.shape[1]),dtype=np.int8))
    U_B_matrix=sp.bmat([[u_u,U_B_graph],
                        [U_B_graph.T,b_b]])

    mat1=sp.hstack((u_u,U_B_graph,U_I_graph))
    mat2=sp.hstack((U_B_graph.T,b_b,B_I_graph))
    mat3=sp.hstack((U_I_graph.T,B_I_graph.T,i_i))
    U_B_I_graph=sp.vstack((mat1,mat2,mat3)).tocsr()
    #
    print("U_B_I_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(U_B_I_graph)))

    # U_I_matrix=sp.bmat([[u_u,U_I_graph],
    #                     [U_I_graph.T,i_i]])
    #
    # print("U_I_graph",sample_hyperbolicity(U_I_matrix))
    #
    # B_I_matrix=sp.bmat([[b_b,B_I_graph],
    #                     [B_I_graph.T,i_i]])
    # print("B_I_graph",sample_hyperbolicity(B_I_matrix))



def getHyperbolicityOfSample_Dataset(path='../data/', DatasetName='Youshu', fileName='user_bundle_train.txt',sample_Num=500,deleteBI=True):
    with open(os.path.join(path, DatasetName, '{}_data_size.txt'.format(DatasetName)), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    print(num_users , num_items , num_bundles)
    if sample_Num>num_users:
        sample_Num=num_users
    print(sample_Num)
    indice, values = loadDataset(path, DatasetName, fileName)

    sample_user=np.random.choice(np.arange(num_users),sample_Num,replace=False)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    U_B_graph = coo_matrix((values, (rowIndice, colIndice)), shape=(num_users, num_bundles)).tocsr().astype(np.int8)
    U_B_graph=U_B_graph[sample_user,:]

    user_indices,bundle_indices=np.nonzero(U_B_graph)
    bundle_indices=np.unique(bundle_indices)
    U_B_graph=U_B_graph[:, bundle_indices]

    fileName = 'user_item.txt'
    U_I_indice, u_i_values = loadDataset(path, DatasetName, fileName)
    U_I_rowIndice = U_I_indice[:, 0]
    U_I_colIndice = U_I_indice[:, 1]
    U_I_graph = csr_matrix((u_i_values, (U_I_rowIndice, U_I_colIndice)), shape=(num_users, num_items)).tocsr().astype(np.int8)
    U_I_graph=U_I_graph[sample_user,:]

    _, item_indices = np.nonzero(U_I_graph)
    item_indices=np.unique(item_indices)
    U_I_graph = U_I_graph[:, item_indices]
    fileName = 'bundle_item.txt'
    B_I_indice, b_i_values = loadDataset(path, DatasetName, fileName)
    B_I_rowIndice = B_I_indice[:, 0]
    B_I_colIndice = B_I_indice[:, 1]

    B_I_graph = csr_matrix((b_i_values, (B_I_rowIndice, B_I_colIndice)), shape=(num_bundles, num_items)).tocsr().astype(np.int8)
    B_I_graph=B_I_graph[bundle_indices,:]
    B_I_graph = B_I_graph[:, item_indices]

    if deleteBI:
        B_I_graph = np.zeros((B_I_graph.shape[0],B_I_graph.shape[1]),dtype=np.int8)
    u_u=csr_matrix(np.zeros((U_B_graph.shape[0],U_B_graph.shape[0]),dtype=np.int8))
    b_b=csr_matrix(np.zeros((B_I_graph.shape[0],B_I_graph.shape[0]),dtype=np.int8))
    i_i=csr_matrix(np.zeros((B_I_graph.shape[1],B_I_graph.shape[1]),dtype=np.int8))

    mat1=sp.hstack((u_u,U_B_graph,U_I_graph))
    mat2=sp.hstack((U_B_graph.T,b_b,B_I_graph))
    mat3=sp.hstack((U_I_graph.T,B_I_graph.T,i_i))
    U_B_I_graph=sp.vstack((mat1,mat2,mat3)).tocsr()
    #
    print("U_B_I_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(U_B_I_graph)))

def getHyperbolicityOfSample_Dataset(path='../data/', DatasetName='Youshu', fileName='user_bundle_train.txt',sample_Num=500,deleteBI=True):
    with open(os.path.join(path, DatasetName, '{}_data_size.txt'.format(DatasetName)), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    print(num_users , num_items , num_bundles)
    if sample_Num>num_users:
        sample_Num=num_users
    print(sample_Num)
    indice, values = loadDataset(path, DatasetName, fileName)

    sample_user=np.random.choice(np.arange(num_users),sample_Num,replace=False)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    U_B_graph = coo_matrix((values, (rowIndice, colIndice)), shape=(num_users, num_bundles)).tocsr().astype(np.int8)
    U_B_graph=U_B_graph[sample_user,:]

    user_indices,bundle_indices=np.nonzero(U_B_graph)
    bundle_indices=np.unique(bundle_indices)
    U_B_graph=U_B_graph[:, bundle_indices]

    fileName = 'user_item.txt'
    U_I_indice, u_i_values = loadDataset(path, DatasetName, fileName)
    U_I_rowIndice = U_I_indice[:, 0]
    U_I_colIndice = U_I_indice[:, 1]
    U_I_graph = csr_matrix((u_i_values, (U_I_rowIndice, U_I_colIndice)), shape=(num_users, num_items)).tocsr().astype(np.int8)
    U_I_graph=U_I_graph[sample_user,:]

    _, item_indices = np.nonzero(U_I_graph)
    item_indices=np.unique(item_indices)
    U_I_graph = U_I_graph[:, item_indices]
    fileName = 'bundle_item.txt'
    B_I_indice, b_i_values = loadDataset(path, DatasetName, fileName)
    B_I_rowIndice = B_I_indice[:, 0]
    B_I_colIndice = B_I_indice[:, 1]

    B_I_graph = csr_matrix((b_i_values, (B_I_rowIndice, B_I_colIndice)), shape=(num_bundles, num_items)).tocsr().astype(np.int8)
    B_I_graph=B_I_graph[bundle_indices,:]
    B_I_graph = B_I_graph[:, item_indices]

    if deleteBI:
        B_I_graph = np.zeros((B_I_graph.shape[0],B_I_graph.shape[1]),dtype=np.int8)
    u_u=csr_matrix(np.zeros((U_B_graph.shape[0],U_B_graph.shape[0]),dtype=np.int8))
    b_b=csr_matrix(np.zeros((B_I_graph.shape[0],B_I_graph.shape[0]),dtype=np.int8))
    i_i=csr_matrix(np.zeros((B_I_graph.shape[1],B_I_graph.shape[1]),dtype=np.int8))

    mat1=sp.hstack((u_u,U_B_graph,U_I_graph))
    mat2=sp.hstack((U_B_graph.T,b_b,B_I_graph))
    mat3=sp.hstack((U_I_graph.T,B_I_graph.T,i_i))
    U_B_I_graph=sp.vstack((mat1,mat2,mat3)).tocsr()
    #
    print("U_B_I_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(U_B_I_graph)))

def getUBHyperbolicity(path='../data',datasetName='Youshu',sample_Num=5000,fileName='Youshu_data_size.txt'):
    with open(os.path.join(path,datasetName, fileName), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    print(num_users, num_items, num_bundles)
    indice, values = loadDataset(path, datasetName, 'user_bundle_train.txt')

    sample_user = np.random.choice(np.arange(num_users), sample_Num, replace=False)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    U_B_graph = coo_matrix((values, (rowIndice, colIndice)), shape=(num_users, num_bundles)).tocsr().astype(np.int8)
    U_B_graph = U_B_graph[sample_user, :]

    user_indices, bundle_indices = np.nonzero(U_B_graph)
    bundle_indices = np.unique(bundle_indices)
    U_B_graph = U_B_graph[:, bundle_indices]
    u_u=csr_matrix(np.zeros((U_B_graph.shape[0],U_B_graph.shape[0]),dtype=np.int8))
    b_b=csr_matrix(np.zeros((U_B_graph.shape[1],U_B_graph.shape[1]),dtype=np.int8))
    mat1=sp.hstack((u_u,U_B_graph))
    mat2=sp.hstack((U_B_graph.T,b_b))
    U_B_entire_graph=sp.vstack((mat1,mat2)).tocsr()

    print("U_B_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(U_B_entire_graph)))


def getUIHyperbolicity(path='../data',datasetName='Youshu',sample_Num=5000,fileName='Youshu_data_size.txt'):
    with open(os.path.join(path,datasetName, fileName), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    print(num_users, num_items, num_bundles)
    indice, values = loadDataset(path, datasetName, 'user_item.txt')

    sample_user = np.random.choice(np.arange(num_users), sample_Num, replace=False)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    U_I_graph = coo_matrix((values, (rowIndice, colIndice)), shape=(num_users, num_items)).tocsr().astype(np.int8)
    U_I_graph = U_I_graph[sample_user, :]

    user_indices, item_indices = np.nonzero(U_I_graph)
    item_indices = np.unique(item_indices)
    U_I_graph = U_I_graph[:, item_indices]
    u_u=csr_matrix(np.zeros((U_I_graph.shape[0],U_I_graph.shape[0]),dtype=np.int8))
    i_i=csr_matrix(np.zeros((U_I_graph.shape[1],U_I_graph.shape[1]),dtype=np.int8))
    mat1=sp.hstack((u_u,U_I_graph))
    mat2=sp.hstack((U_I_graph.T,i_i))
    U_I_entire_graph=sp.vstack((mat1,mat2)).tocsr()

    print("U_I_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(U_I_entire_graph)))

def getBIHyperbolicity(path='../data',datasetName='Youshu',sample_Num=5000,fileName='Youshu_data_size.txt'):
    with open(os.path.join(path,datasetName, fileName), 'r') as f:
        num_users, num_bundles, num_items = [int(s) for s in f.readline().split('\t')][:3]

    print(num_users, num_items, num_bundles)
    indice, values = loadDataset(path, datasetName, 'bundle_item.txt')

    sample_bundle = np.random.choice(np.arange(num_bundles), sample_Num, replace=False)

    rowIndice = indice[:, 0]
    colIndice = indice[:, 1]

    B_I = coo_matrix((values, (rowIndice, colIndice)), shape=(num_bundles, num_items)).tocsr().astype(np.int8)
    B_I_graph = B_I[sample_bundle, :]

    user_indices, item_indices = np.nonzero(B_I_graph)
    item_indices = np.unique(item_indices)
    B_I_graph = B_I_graph[:, item_indices]
    b_b=csr_matrix(np.zeros((B_I_graph.shape[0],B_I_graph.shape[0]),dtype=np.int8))
    i_i=csr_matrix(np.zeros((B_I_graph.shape[1],B_I_graph.shape[1]),dtype=np.int8))
    mat1=sp.hstack((b_b,B_I_graph))
    mat2=sp.hstack((B_I_graph.T,i_i))
    B_I_entire_graph=sp.vstack((mat1,mat2)).tocsr()

    print("B_I_graph",hyperbolicity_sample(networkx.from_scipy_sparse_matrix(B_I_entire_graph)))

if __name__ == '__main__':


    # getUBHyperbolicity(path='../data',datasetName='Youshu',sample_Num=8039,fileName='Youshu_data_size.txt')
    # getUIHyperbolicity(path='../data', datasetName='Youshu', sample_Num=8039, fileName='Youshu_data_size.txt')
    # getBIHyperbolicity(path='../data', datasetName='Youshu', sample_Num=4771, fileName='Youshu_data_size.txt')
    # getHyperbolicityOfDataset(path='../data/', DatasetName='Youshu', fileName='user_bundle_train.txt')
    # getHyperbolicityOfSample_Dataset(DatasetName='Youshu',sample_Num=8039, fileName='user_bundle_train.txt',deleteBI=False)
    getUBHyperbolicity(path='../data',datasetName='NetEase',sample_Num=18528,fileName='NetEase_data_size.txt')
    getUIHyperbolicity(path='../data', datasetName='NetEase', sample_Num=18528, fileName='NetEase_data_size.txt')
    getBIHyperbolicity(path='../data', datasetName='NetEase', sample_Num=18528, fileName='NetEase_data_size.txt')
    # getHyperbolicityOfDataset(path='../data/', DatasetName='NetEase', fileName='user_bundle_train.txt')
    # getHyperbolicityOfSample_Dataset(DatasetName='NetEase', sample_Num=18528, fileName='user_bundle_train.txt', deleteBI=False)
    # sampleNumList=[50,500,5000,10000,18528]
    # for sampleNum in sampleNumList:
    #     fileName='user_bundle_test.txt'
    #     getHyperbolicityOfSample_Dataset(DatasetName='Youshu',sample_Num=sampleNum, fileName=fileName,deleteBI=True)
    #     print("not delete B-I interaction")
    #     getHyperbolicityOfSample_Dataset(DatasetName='Youshu', sample_Num=sampleNum,  fileName=fileName,deleteBI=False)
