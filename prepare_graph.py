# -*- coding: utf-8 -*-
# @Author  : zongpei ma
import scanpy as sc
import numpy as np
import pandas as pd
import  os

#compute correlation for cell
def correlation(data_numpy, k, corr_type='pearson'):

    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors

#prepare graph knn or pknn
def prepare_graphs(adata_khvg, dataset_name, save_path, args):

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

# do load graph
def load_graph(edge_path):
    edgelist = []
    with open(edge_path, 'r') as edge_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edge_file.readlines()]
    return edgelist  # [(0, 1), (0, 2), (1, 3)]
