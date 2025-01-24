# -*- coding: utf-8 -*-
# @Author  : zongpei ma
import scanpy as sc
import numpy as np
import pandas as pd
import  os
from config import Args



def correlation(data_numpy, k, corr_type='pearson'):
    """
    计算数据的相关性，并返回相关性矩阵和每个点的前k个最相关的邻居。
    :param data_numpy: 基因表达矩阵
    :param k: 多少个邻居
    :param corr_type:
    :return:
    corr：相关性矩阵
    neighbors： 邻居矩阵，shape为 num_cell*k's neighbors
    """
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors

#
# adata=sc.read_h5ad('data/preprocessed/yan_preprocessed.h5ad')
# print(adata.X.shape)
# corr,neighbors=correlation(adata.X,40)
# print(corr.shape)
# print(neighbors.shape)
#




def prepare_graphs(adata_khvg, dataset_name, save_path, args):
    """
    :param adata_khvg: anndata对象
    :param dataset_name: 数据集的名称用于保存
    :param save_path: 保存文件   os.path.join(os.path.dirname(__file__), 'graph')
    :param args: 从config.py里面导入
    :return:
    """
    if args.graph_type == 'KNN':
        print('Computing KNN graph by scanpy...')
        # use package scanpy to compute knn graph
        distances_csr_matrix = \
            sc.pp.neighbors(adata_khvg, n_neighbors=args.k + 1, knn=True, copy=True).obsp[
                'distances']
        # ndarray
        distances = distances_csr_matrix.A
        # resize
        neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], args.k))

    elif args.graph_type == 'PKNN':
        print('Computing PKNN graph...')
        if isinstance(adata_khvg.X, np.ndarray):
            X_khvg = adata_khvg.X
        else:
            X_khvg = adata_khvg.X.toarray()
        distances, neighbors = correlation(data_numpy=X_khvg, k=args.k + 1)

    if args.graph_distance_cutoff_num_stds:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(args.graph_distance_cutoff_num_stds) * np.std(
            np.nonzero(distances), axis=None)
    # shape: 2 * (the number of edge)
    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if args.graph_distance_cutoff_num_stds:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    # save
    if args.save_graph:
        num_hvg = adata_khvg.shape[1]
        k_file = args.k
        if args.graph_type == 'KNN':
            graph_name = 'Scanpy'
        elif args.graph_type == 'PKNN':
            graph_name = 'Pearson'

        filename = f'{dataset_name}_{graph_name}_KNN_K{k_file}_gHVG_{num_hvg}.txt'

        final_path = os.path.join(save_path, filename)
        print(f'Saving graph to {final_path}...')
        with open(final_path, 'w') as f:
            edges = [' '.join(e) + '\n' for e in edgelist]
            f.writelines(edges)

    return edgelist






def load_graph(edge_path):
    edgelist = []
    with open(edge_path, 'r') as edge_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edge_file.readlines()]
    return edgelist  # [(0, 1), (0, 2), (1, 3)]
