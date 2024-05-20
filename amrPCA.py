# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:59:35 2024

@author: Alejandro
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


matrix_file = "C:/Users/Alejandro/Desktop/Data/amr-count.tsv"
df = pd.read_csv(matrix_file, sep='\t', index_col=0)  # Assuming ARO identifiers are in the first column and used as row index


pca = PCA()
pca.fit(df)


pca_data = pca.transform(df)


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.show()
