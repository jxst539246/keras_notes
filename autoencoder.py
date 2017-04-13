from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
encoding_size = 32

img_input = Input(shape=(28*28,))
encoded = Dense(encoding_size,activation='relu')(img_input)
decoded = Dense(28*28,activation='sigmoid')(encoded)
autoencoder = Model(img_input,decoded)

encoder = Model(img_input,encoded)
encoded_input = Input(shape=(encoding_size,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
(x_train,_),(x_test,_) = mnist.load_data()



