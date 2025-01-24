

# -*- coding: utf-8 -*-
# @Author  : zongpei ma
import argparse
import math
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.cluster import KMeans
import umap
import hdbscan
import loader
import layer
import scanpy as sc
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from prepare_graph import prepare_graphs, load_graph
from config import Args
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('model training use', device)


args = Args(
    graph_type='KNN',
    k=10,
    graph_distance_cutoff_num_stds=0.0,
    save_graph=True,
    save_graph_path='graph/scMGCL',
    batch_size=10000000,  # 一定要设置无限大
    workers=0,
    alpha=0.3,  # 0.30.91
    beta=0.1,  # 0.1
    cluster_name='kmeans',
    seed=2,
    epochs=700,  # 700
    cluster_num=6
)

def adjust_learning_rate(optimizer, epoch, args):
    lr = 0.1
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["ARI"] = ARI(y_true, y_pred)
    metrics["NMI"] = NMI(y_true, y_pred)
    return metrics

from sklearn.impute import SimpleImputer

def cluster(embedding, gt_labels, num_cluster, args):
    # 使用SimpleImputer替换NaN值
    imputer = SimpleImputer(strategy='mean')
    embedding = imputer.fit_transform(embedding)

    if args.cluster_name == 'kmeans':
        pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embedding).labels_
        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
        return eval_supervised_metrics, pd_labels

    if args.cluster_name == 'hdbscan':
        # UMAP降维
        umap_reducer = umap.UMAP()
        u = umap_reducer.fit_transform(embedding)
        cl_sizes = [10, 25, 50, 100]
        min_samples = [5, 10, 25, 50]
        hdbscan_dict = {}
        ari_dict = {}
        for cl_size in cl_sizes:
            for min_sample in min_samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cl_size, min_samples=min_sample)
                clusterer.fit(u)
                ari_dict[(cl_size, min_sample)] = compute_metrics(gt_labels, clusterer.labels_)
                hdbscan_dict[(cl_size, min_sample)] = clusterer.labels_
        max_tuple = max(ari_dict, key=lambda x: ari_dict[x]['ARI'])
        return ari_dict[max_tuple], hdbscan_dict[max_tuple]


input_h5ad_path = 'data/preprocessed/yan_preprocessed.h5ad'
processed_adata = sc.read_h5ad(input_h5ad_path)
label_col_name = 'x'
pre_path, filename = os.path.split(input_h5ad_path)
dataset_name, ext = os.path.splitext(filename)

edgelist = prepare_graphs(processed_adata, dataset_name, args.save_graph_path, args)

num_nodes = processed_adata.shape[0]
num_gene = processed_adata.shape[1]
print('基因数量', num_gene)
print(f'The graph has {len(edgelist)} edges.')
edge_index = np.array(edgelist).astype(int).T
edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)

# 确保数据类型一致，将 processed_adata.X 转为 float32
processed_adata.X = processed_adata.X.astype(np.float32)

global_graph = Data(x=torch.tensor(processed_adata.X, dtype=torch.float32), edge_index=edge_index)
global_graph.n_id = torch.arange(global_graph.num_nodes)


train_dataset = loader.Matrix(
    adata=processed_adata,
    global_graph=global_graph,
    obs_label_colname=label_col_name,
    augmentation=False
)

eval_dataset = loader.Matrix(
    adata=processed_adata,
    global_graph=global_graph,
    obs_label_colname=label_col_name,
    augmentation=False
)

if train_dataset.num_cells < args.batch_size:
    args.batch_size = train_dataset.num_cells

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=num_nodes, shuffle=False,
    sampler=None, num_workers=args.workers, pin_memory=True
)

print("creating model 'min-max-model' ")
model = layer.Net(num_gene=num_gene, latent_dim=128, lower_dim=64, sigma=0.01, cluster_num=args.cluster_num).to(device)  # 128 64   0.01 256-128
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)  # 0.01,weight_decay=5e-4

def loss_infomax(x, x_cl):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_cl_abs = x_cl.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / sim_matrix.sum(dim=1)
    loss = -torch.log(loss).mean()
    return loss

def loss_infomin(x, x_cl):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_cl_abs = x_cl.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = torch.log(loss + 1).mean()
    return loss

def inference(eval_loader, model, global_graph):
    print('Inference...')
    model.eval()
    features = []
    labels = []

    for i, (images, index, label) in enumerate(eval_loader):
        images = images.to(device)
        with torch.no_grad():
            feat, _, _, _, _, _ = model(images, global_graph.edge_index.to(device))
        feat_pred = feat.data.cpu().numpy()
        label_true = label

        features.append(feat_pred)
        labels.append(label_true)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels

best_ari = 0
best_nmi = 0
features_best = None

for epoch in range(0, args.epochs):
    for i, (matrix, index, label) in enumerate(train_loader):
        matrix = matrix.to(device)
        global_graph1 = global_graph.edge_index.to(device)
        gx, gx1, gx2, loss_graph, gxc, gx1_c = model(matrix, global_graph1)

        loss1 = loss_infomax(gx, gx1)
        loss2 = loss_infomax(gx, gx2)
        loss3 = loss_infomin(gx1, gx2)
        loss_cluster = loss_infomax(gxc, gx1_c)

        loss = args.alpha * loss1 + (1 - args.alpha) * loss2 + args.beta * loss3 + 0.3 * loss_graph + 0.5 * loss_cluster

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        features, labels = inference(eval_loader, model, global_graph)
        eval_supervised_metrics, pd_labels = cluster(features, labels, args.cluster_num, args)
        ari, nmi = eval_supervised_metrics["ARI"], eval_supervised_metrics['NMI']
        print(ari, nmi, 'epoch:', epoch)
        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi
            features_best = features
            np.savetxt('best_embedding.txt', features_best, delimiter=',')

            with open('best.txt', 'w') as file:
                print('在当前epoch保存最佳embedding')
                file.write(f'Highest ARI: {best_ari}\n')
                file.write(f'Highest NMI: {best_nmi}\n')
