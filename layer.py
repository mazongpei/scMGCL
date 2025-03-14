# -*- coding: utf-8 -*-
# @Author  : zongpei ma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, PReLU
from torch_geometric.nn import GINConv, global_add_pool,GATConv
import numpy as np
from torch.nn import Linear

class GATEncoder(nn.Module):

    def __init__(self, num_genes, latent_dim, num_heads=10
                 , dropout=0.2):
        super(GATEncoder, self).__init__()
        # initialize parameter
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.gat_layer_1 = GATConv(
            in_channels=num_genes, out_channels=128,
            heads=num_heads,
            dropout=dropout,
            concat=True)
        in_dim2 = 128 * num_heads
        self.gat_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=latent_dim,
            heads=num_heads,
            concat=False)

    def forward(self, x, edge_index):
        hidden_out1 = self.gat_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)

        hidden_out2 = self.gat_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        
        return hidden_out2

class MLP(nn.Module):
    def __init__(self, num_gene, latent_dim):
        super(MLP, self).__init__()
        self.num_gene = num_gene
        self.latent_dim = latent_dim

        self.layer1 = Linear(num_gene, latent_dim)

        self.layer2 = Linear(latent_dim, latent_dim)

    def forward(self, x):

        out1 = self.layer1(x)
        out1 = F.relu(out1)

        embedding = self.layer2(out1)
        return embedding

class Projection_head(nn.Module):

    def __init__(self,latent_dim,lower_dim):
        super(Projection_head,self).__init__()
        self.latent_dim=latent_dim
        self.lower_dim=lower_dim
        self.fc=Linear(latent_dim,lower_dim)

    def forward(self,x):
        out=self.fc(x)

        return out

class Net(nn.Module):

    def __init__(self,num_gene,latent_dim,lower_dim,sigma,cluster_num):#alpha,beta
        super(Net,self).__init__()
        self.num_gene=num_gene
        self.latent_dim=latent_dim
        self.lower_dim=lower_dim
        self.sigma=sigma
        self.cluster_num=cluster_num
        self.GAT=GATEncoder(num_gene,latent_dim)
        self.MLP=MLP(num_gene,latent_dim)
        self.FC=Projection_head(latent_dim,lower_dim)
        self.FC_cluster=Projection_head(latent_dim,cluster_num)

    def global_avg_pooling(self,x):
        return torch.mean(x, dim=0)
        
    def loss_graph_polling(self,gx, gx1, gx2):
        fenzi = torch.exp(torch.matmul(gx, gx1.T)/0.2)
        fenmu = torch.exp(torch.matmul(gx, gx1.T)/0.2) + torch.exp(torch.matmul(gx, gx2.T)/0.2) + torch.exp(
            torch.matmul(gx1, gx2.T)/0.2)
        loss = fenzi / fenmu
        loss = -torch.log10(loss)
        return  loss.mean()

    def forward(self,x,index):
        if x is None:
            x = torch.ones((x.shape[0], 1))
        x_aug = x + F.normalize(torch.normal(0, torch.ones_like(x) * self.sigma), dim=1)

        gx=self.GAT(x,index)
        gx1=self.GAT(x_aug,index)
        gx2=self.MLP(x)

        graph_origal = self.global_avg_pooling(gx)
        graph_aug = self.global_avg_pooling(gx1)
        graph_no = self.global_avg_pooling(gx2)
        loss_graph=self.loss_graph_polling(graph_origal,graph_aug,graph_no)

        gx_node=self.FC(gx)
        gx1_node=self.FC(gx1)
        gx2_node=self.FC(gx2)

        gx_c=self.FC_cluster(gx)
        gx1_c=self.FC_cluster(gx1)

        return gx_node,gx1_node,gx2_node,loss_graph,gx_c.T,gx1_c.T

    def loss_infomax(self, x, x_cl):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_cl_abs = x_cl.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

    def loss_infomin(self, x, x_cl):
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


