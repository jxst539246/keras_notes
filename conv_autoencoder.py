from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras import  regularizers
from keras.callbacks import TensorBoard
import numpy as np
encoding_size = 32

img_input = Input(shape=(28,28,1))

x = Conv2D(16,3,3,activation='relu',border_mode='same')(img_input)
x = MaxPooling2D((2,2),border_mode='same')(x)
x = Conv2D(8,3,3,activation='relu',border_mode='same')(x)
x = MaxPooling2D((2,2),border_mode='same')(x)
x = Conv2D(8,3,3,activation='relu',border_mode='same')(x)
encoded = MaxPooling2D((2,2), border_mode='same')(x)
# output (4,4,8)

x = Conv2D(8,3,3,activation='relu',border_mode='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8,3,3,activation='relu',border_mode='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16,3,3,activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,3,3,activation='sigmoid',border_mode='same')(x)

autoencoder = Model(img_input,decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
#print(autoencoder.summary())
#print(decoder.summary())
#print(encoder.summary())

(x_train,_),(x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),28,28,1))
x_test = x_test.reshape((len(x_test),28,28,1))

autoencoder.fit(x_train,x_train,
                batch_size=64,
                nb_epoch=20,
                validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir="/tmp/autoencoder")])

decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()