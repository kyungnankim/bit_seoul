import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


(x_train, _), (x_test, _) =mnist.load_data()
x = np.append(x_train, x_test, axis = 0)
print(x.shape) # (70000, 28, 28)

pca = PCA(7) 
x2d = pca.fit_transform(x)
print(x2d.shape)

pca_EVR = pca.explained_variance_ratio_ 
print(pca_EVR)
print(sum(pca_EVR))

pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
print('선택할 차원 수 :', d)