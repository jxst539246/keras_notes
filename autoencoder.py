from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras import  regularizers
import numpy as np
encoding_size = 32

img_input = Input(shape=(28*28,))
encoded_input = Input(shape=(encoding_size,))
encoded = Dense(128,activation='relu')(img_input)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(32,activation='relu')(encoded)
decoded = encoded
decoder = encoded_input
layer = Dense(64,activation='relu')
decoder = layer(decoder)
decoded = layer(decoded)
out_layer = Dense(28*28,activation='sigmoid')

decoded = out_layer(decoded)
decoder = out_layer(decoder)

autoencoder = Model(img_input,decoded)
encoder = Model(img_input,encoded)
final_decoder = Model(encoded_input,decoder)

print(final_decoder.summary())


autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
#print(autoencoder.summary())
#print(decoder.summary())
#print(encoder.summary())

(x_train,_),(x_test,_) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),784))
x_test = x_test.reshape((len(x_test),784))

autoencoder.fit(x_train,x_train,batch_size=64,nb_epoch=20,validation_data=(x_test,x_test))
encoded_img = encoder.predict(x_test)
decoded_img = final_decoder.predict(encoded_img)

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