import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from keras.models import Sequential
from keras.layers.core import  Dense,Activation
from keras.utils import np_utils

iris = sns.load_dataset('iris')
X = iris.values[:,:4]
Y = iris.values[:,4]
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,train_size=0.5,random_state=0)
lr = LogisticRegressionCV()
lr.fit(train_X,train_Y)
print("Accuracy = {:.2f}".format(lr.score(test_X, test_Y)))

def one_hot_encode(arr):
    uniques, ids = np.unique(arr,return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

train_Y_ohe = one_hot_encode(train_Y)
test_Y_ohe = one_hot_encode(test_Y)
model = Sequential()
model.add(Dense(16,input_shape=(4,),activation='sigmoid'))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_X,train_Y_ohe,batch_size=1,nb_epoch=50)
loss,accuracy=model.evaluate(test_X,test_Y_ohe)
print("\nAccuracy = {:.2f}".format(accuracy))

