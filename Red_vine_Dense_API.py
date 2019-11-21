import os
os.environ['PYTHONHASHSEED'] = str(0)

#%%
import numpy as np

np.random.seed(2)

# загружаем данные с фичами
dataset = np.loadtxt("/home/krisantis/Desktop/Vine/red/winequality-red.txt",
                     delimiter=";")

X = dataset[:, 0:-1]
X = np.asarray(X).astype('float32')
Y = dataset[:, -1:]
Y = np.asarray(Y).astype('int')


def to_one_hot(labels, demension=10):
    results = np.zeros((len(labels), demension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


Y = to_one_hot(Y)

#Перемешивание вариантов
indices = np.arange(dataset.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

#нормализация
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

#X1 = np.column_stack((X[:,1],X[:,5],X[:,6],X[:,8],X[:,10]))   #2,6,7,9,11    3,5,8,9,10
#X2 = np.column_stack((X[:,0],X[:,2],X[:,3],X[:,4],X[:,7]))    #1,3,4,5,8     2,5,6,7,9
#X3 = np.column_stack((X[:,2],X[:,3],X[:,7],X[:,8],X[:,9]))   #3,4,8,9,10     1,4,5,10,11

#X1 = np.column_stack((X[:,2],X[:,4],X[:,7],X[:,8],X[:,9]))   #3,5,8,9,10
#X2 = np.column_stack((X[:,1],X[:,4],X[:,5],X[:,6],X[:,8]))    #2,5,6,7,9
#X3 = np.column_stack((X[:,0],X[:,3],X[:,4],X[:,9],X[:,10]))   #1,4,5,10,11

X1 = np.column_stack(
    (X[:, 2], X[:, 4], X[:, 7], X[:, 8], X[:, 9]))  #3,5,8,9,10
X2 = np.column_stack((X[:, 1], X[:, 5], X[:, 6]))  #2,5,6,7,9
X3 = np.column_stack((X[:, 0], X[:, 3], X[:, 10]))  #1,4,5,10,11
#начинаем формировать апи сеть

#начинаем формировать апи сеть

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.models import Model
from keras import layers
from keras import Input

#--------------- digit-2,6,7,9,11-----------------
digit_input_1 = Input(shape=(X1.shape[1], ))
dense_digit_layer_1 = layers.Dense(64, activation='relu')(digit_input_1)
dense_digit_layer_1 = layers.Dense(32, activation='relu')(dense_digit_layer_1)
dense_digit_layer_1 = layers.Dense(16, activation='relu')(dense_digit_layer_1)

#--------------- digit-1,3,4,5,8-----------------
digit_input_2 = Input(shape=(X2.shape[1], ))
dense_digit_layer_2 = layers.Dense(64, activation='relu')(digit_input_2)
dense_digit_layer_2 = layers.Dense(32, activation='relu')(dense_digit_layer_2)
dense_digit_layer_2 = layers.Dense(16, activation='relu')(dense_digit_layer_2)

#--------------- digit-,4,8,9,10-----------------
digit_input_3 = Input(shape=(X3.shape[1], ))
dense_digit_layer_3 = layers.Dense(64, activation='relu')(digit_input_3)
dense_digit_layer_3 = layers.Dense(32, activation='relu')(dense_digit_layer_3)
dense_digit_layer_3 = layers.Dense(16, activation='relu')(dense_digit_layer_3)

#--------------- concatenated-----------------
concatenated = layers.concatenate(
    [dense_digit_layer_1, dense_digit_layer_2, dense_digit_layer_3], axis=-1)

conc_layrs = layers.Dense(16, activation='relu')(concatenated)
conc_layrs = layers.Dense(16, activation='relu')(conc_layrs)
answer = layers.Dense(10, activation='softmax')(conc_layrs)

model = Model([digit_input_1, digit_input_2, digit_input_3], answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit([X1, X2, X3],
                    Y,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.3)

#графики изменения качества модели

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()