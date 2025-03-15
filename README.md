# **scMGCL: Multi-Level Graph Contrastive Learning for scRNA-Seq Data Analysis**

## Introduction

 In this study, we propose a novel multi-level graph contrastive learning method, **scMGCL**. Through data augmentation, we obtained three views: the original view, the topology augmentation view, and the semantic augmentation view. It designs a min-max principle to constrain mutual information across instance, cluster, and global levels, including the original view and its augmentation views, and between different augmentation views. In the instance-level contrastive learning module, cell representations are learned. In the cluster-level contrastive learning module, cluster representations are obtained. Finally, at the global level, global representations of dataset are derived. Together, these contrastive learning modules collaboratively enhance the representation learning across cells, clusters, and the entire dataset. Extensive experiments are conducted on 11 scRNA-seq datasets and experimental results show that scMGCL achieves better or competitive performance compared with state-of-the-art methods. In addition, scMGCL can effectively remove batch effects, accurately analyzes marker genes, and identifies cell developmental trajectories, demonstrating its biological effectiveness.

## Installation

The package can be installed by `git clone https://github.com/mazongpei/scMGCL.git`. The testing setup involves a Windows operating system with 16GB of RAM, powered by an NVIDIA GeForce RTX 4070 Ti GPU and an Intel(R) Core(TM) i5-13490f CPU @ 4..80GHz.

### Utilize a virtual environment using Anaconda

You can set up the primary environment for scMGCL by using the following command:

```
conda env create -f environment.yml
```

## Running scMGCL

### 1. Data Preprocessing

It supports three types of input file formats: **H5AD, H5, and CSV data**. Throughout the preprocessing procedure, there are a total of five operations, encompassing cell-gene filtering, normalization, logarithmic transformation, scaling, and the selection of highly variable genes. 

```python
# H5AD files
python preprocess/preprocess_data.py --input_h5ad_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
# H5 files
python preprocess/preprocess_data.py --input_h5_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
# CSV files
python preprocess/preprocess_data.py --count_csv_path=Path_to_input --label_csv_path=Path_to_input --save_h5ad_dir=Path_to_save --filter --norm --log --scale --select_hvg
```

### 2. Apply scMGCL

By utilizing the preprocessed input data, you have the option to invoke the subsequent script for executing the scMGCL method:

```python
python main.py  --input_h5ad_path "data/preprocessed/XXX.h5ad" --cluster_num XX --graph_type xx --cluster_name xx
```

## Running example

### 1. Collect Dataset.

All data used in this study were obtained from publicly available datasets. Our sample dataset is stored in the directory "data/original/deng.h5".

### 2. Generate Preprocessed H5AD File.

```python
python preprocess/preprocess_data.py --input_h5_path="./data/original/deng.h5" --save_h5ad_dir="./data/preprocessed/" --filter --norm --log --scale --select_hvg
```

### 3. Apply scMGCL

```python
python main.py --input_h5ad_path="data/preprocessed/deng_preprocessed.h5ad" --cluster_num 6 --graph_type KNN --cluster_name kmeans
```
