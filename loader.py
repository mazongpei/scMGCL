# -*- coding: utf-8 -*-
# @Author  : zongpei ma
from anndata._core.anndata import AnnData
from torch.utils.data import Dataset
import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import warnings
warnings.filterwarnings('ignore')

class Matrix(Dataset):
    def __init__(self,
                 adata: AnnData = None,
                 global_graph: Data = None,
                 obs_label_colname: str = "x",
                 ):

        super().__init__()

        self.adata = adata

        if isinstance(self.adata.X, np.ndarray):
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()

        if self.adata.obs.get(obs_label_colname) is not None:
            self.label = self.adata.obs[obs_label_colname]
            self.unique_label = list(set(self.label))
            #  str:number
            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            # number:str
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        else:
            self.label = None

        self.num_cells, self.num_genes = self.adata.shape
        self.global_graph = global_graph

    # a cell
    def __getitem__(self, index):

        sample = self.data[index]

        if self.label is not None:
            label = self.label_encoder[self.label[index]]
        else:
            label = -1

        return sample, index, label

    def __len__(self):
        return self.adata.X.shape[0]

def random_substitution(x):
    random_cell = np.random.randint(x.shape)
    return x[random_cell]
