import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.utils import np_utils

num_classes = 10
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
