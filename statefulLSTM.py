import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

np.random.seed(10)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_int = dict((c,i) for i,c in enumerate(alphabet))
int_to_char = dict((i,c) for i,c in enumerate(alphabet))

seq_length = 1
dataX,dataY = [],[]
for i in range(0,len(alphabet)-seq_length,1):
    #print(i)
    seq_in = alphabet[i:i+seq_length]
    seq_out = alphabet[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in,'->',seq_out)

X = pad_sequences(dataX, maxlen=seq_length, dtype='float32')
# reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], seq_length, 1))
X = X / float(len(alphabet))
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(32,input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(Y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])
model.fit(X,Y,nb_epoch=500,batch_size=1,verbose=2)
scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)
