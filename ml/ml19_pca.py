import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA(9) 
x2d = pca.fit_transform(x)
print(x2d.shape)

pca_EVR = pca.explained_variance_ratio_ 
print(pca_EVR)
print(sum(pca_EVR))
