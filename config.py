# -*- coding: utf-8 -*-
# @Author  : zongpei ma
class Args:
    def __init__(self, graph_type, k, graph_distance_cutoff_num_stds,save_graph,save_graph_path,batch_size,workers,alpha,beta,cluster_name,seed,epochs,cluster_num):
        #保存图所用到的参数
        self.graph_type =graph_type#构件图时用哪种算法，默认为KNN
        self.k =k#KNN中K的个数
        self.graph_distance_cutoff_num_stds =graph_distance_cutoff_num_stds#默认为0
        self.save_graph =save_graph#是否保存图形文件的布尔值
        self.save_graph_path=save_graph_path# 图的存放路径


        #训练所用超参数
        self.batch_size=batch_size
        self.workers=workers
        self.epochs=epochs

        #loss 的参数
        self.alpha=alpha
        self.beta=beta

        #至少包含cluster_name（指定使用的聚类算法名）和seed（随机种子，用于KMeans）等字段
        #聚类指标的时候用的
        self.cluster_name=cluster_name
        self.seed=seed


        #簇的个数
        self.cluster_num=cluster_num