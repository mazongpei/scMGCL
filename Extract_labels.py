#提取label

import scanpy as sc

# 加载 h5 文件
adata = sc.read('data/preprocessed/tmsfpoa-Limb_Muscle_preprocessed.h5ad')

# 提取label信息，假设第一列就是label
cell_labels = adata.obs['x']

# 打印前几行label信息
print("Extracted cell labels:")
print(cell_labels.head())

cell_labels.to_csv('Muscle.csv', index=False)
print("Cell labels have been saved to 'cell_labels.csv'.")

print(adata.X.shape)