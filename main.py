# -*- coding: utf-8 -*-
# @Author  : zongpei ma
import argparse
import math
import os
from sre_parse import parse
from sklearn.impute import SimpleImputer
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
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('model training use', device)
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run scMGCL model training with configurable parameters.')

    parser.add_argument('--graph_type', type=str, default='', choices=['KNN','PKNN'],help='Type of graph to construct')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors for KNN graph')
    parser.add_argument('--graph_distance_cutoff_num_stds', type=float, default=0.0, help='Graph distance cutoff in stds')
    parser.add_argument('--save_graph', type=bool, default=True, help='Whether to save the graph')
    parser.add_argument('--save_graph_path', type=str, default='graph/scMGCL', help='Path to save the graph')
    parser.add_argument('--batch_size', type=int, default=10000000, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for loss_infomax ')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for loss_infomin')
    parser.add_argument('--cluster_name', type=str, default='', choices=['kmeans', 'hdbscan'], help='Clustering method')
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--cluster_num', type=int, default=-1, help='Number of clusters')
    parser.add_argument('--input_h5ad_path', type=str, default="", help='path to input h5ad file')
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--gamma',type=float,default=0.8)
    parser.add_argument('--delta', type=float, default=0.7)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, args):
    lr = 0.01
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["ARI"] = ARI(y_true, y_pred)
    metrics["NMI"] = NMI(y_true, y_pred)
    return metrics

def cluster(embedding, gt_labels, num_cluster, args):

    imputer = SimpleImputer(strategy='mean')
    embedding = imputer.fit_transform(embedding)

    if args.cluster_name == 'kmeans':
        pd_labels = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(embedding).labels_
        eval_supervised_metrics = compute_metrics(gt_labels, pd_labels)
        return eval_supervised_metrics, pd_labels

    if args.cluster_name == 'hdbscan':
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


def main():
    args = parse_arguments()

    processed_adata = sc.read_h5ad(args.input_h5ad_path)
    label_col_name = 'x'
    pre_path, filename = os.path.split(args.input_h5ad_path)
    dataset_name, ext = os.path.splitext(filename)
    edgelist = prepare_graphs(processed_adata, dataset_name, args.save_graph_path, args)
    num_nodes = processed_adata.shape[0]
    num_gene = processed_adata.shape[1]
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    processed_adata.X = processed_adata.X.astype(np.float32)
    global_graph = Data(x=torch.tensor(processed_adata.X, dtype=torch.float32), edge_index=edge_index)
    global_graph.n_id = torch.arange(global_graph.num_nodes)

    train_dataset = loader.Matrix(
        adata=processed_adata,
        global_graph=global_graph,
        obs_label_colname=label_col_name,
    )

    eval_dataset = loader.Matrix(
        adata=processed_adata,
        global_graph=global_graph,
        obs_label_colname=label_col_name,
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

    print("creating model 'scMGCL' ")
    model = layer.Net(num_gene=num_gene, latent_dim=256, lower_dim=128, sigma=0.01, cluster_num=args.cluster_num).to(device)  # 128 64   0.01 256-128
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), args.lr)  # 0.01,weight_decay=5e-4

    def loss_infomax(x, x_cl):
        """
         This function is based on the implementation at:
         https://github.com/xyhappy/GCLMI
         Minor modifications have been made to fit our specific needs.
         """
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
        """
         This function is based on the implementation at:
         https://github.com/xyhappy/GCLMI
         Minor modifications have been made to fit our specific needs.
         """
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
        #adjust_learning_rate(optimizer,epoch, args)
        for i, (matrix, index, label) in enumerate(train_loader):
            matrix = matrix.to(device)
            global_graph1 = global_graph.edge_index.to(device)
            gx, gx1, gx2, loss_graph, gxc, gx1_c = model(matrix, global_graph1)

            loss1 = loss_infomax(gx, gx1)
            loss2 = loss_infomax(gx, gx2)
            loss3 = loss_infomin(gx1, gx2)
            loss_cluster = loss_infomax(gxc, gx1_c)

            loss = args.gamma*(args.alpha * loss1 + (1 - args.alpha) * loss2 + args.beta * loss3) + args.delta*loss_cluster + (1-args.delta)*loss_graph

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            features, labels = inference(eval_loader, model, global_graph)
            eval_supervised_metrics, pd_labels = cluster(features, labels, args.cluster_num, args)
            ari, nmi = eval_supervised_metrics["ARI"], eval_supervised_metrics['NMI']
            if ari > best_ari:
                best_ari = ari
                best_nmi = nmi
                features_best = features
                np.savetxt('best_embedding.txt', features_best, delimiter=',')

                with open('best.txt', 'w') as file:
                    file.write(f'Highest ARI: {best_ari}\n')
                    file.write(f'Highest NMI: {best_nmi}\n')


if __name__ == "__main__":
    main()

