from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #CNN 기본 layer
import matplotlib.pyplot as plt
import numpy as np

# (x_train, y_train),(x_test,y_test) = cifar10.load_data()
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
print(x_train[0])
print("y_train[0]",y_train[0])

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

plt.imshow(x_train[0])
plt.show()