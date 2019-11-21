import numpy as np
 


 
# загружаем данные с фичами
dataset = np.loadtxt("/home/krisantis/Desktop/Vine/red/winequality-red.txt", delimiter=";")


X = dataset[:,0:8]
X_text = dataset[:,8:108]
X=np.asarray(X).astype('float32')
Y = dataset[:,-1:]

indices = np.arange(dataset.shape[0])
np.random.shuffle(indices)
X = X[indices]
X_text = X_text[indices]
Y = Y[indices]

#обучение на выборке из 20000 образцов
training_samples = 1500000
#проверка на выборке из 5000
validation_samples = 500000

X_train = X[:training_samples]
X_text_train = X_text[:training_samples]
Y_train = Y[:training_samples]
X_val = X[training_samples : training_samples + validation_samples]
X_text_val = X_text[training_samples : training_samples + validation_samples]
Y_val = Y[training_samples : training_samples + validation_samples]

#нормализация
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std

X_val -= mean
X_val /= std

#начинаем формировать апи сеть

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.models import Model
from keras import layers
from keras import Input

digit_train_size = X_train.shape[1]
text_train_size = X_text_train.shape[1]
answer_train_size = 1
max_words = 10000
embedding_dim = 300

#--------------- digit-----------------
digit_input = Input(shape=(digit_train_size,))
dense_digit_layer = layers.Dense(64, activation='relu')(digit_input)
dense_digit_layer = layers.Dense(32, activation='relu')(dense_digit_layer)
dense_digit_layer = layers.Dense(32, activation='relu')(dense_digit_layer)

#--------------- text-----------------
text_input = Input(shape=(text_train_size,), dtype='float32')
embidding_text_layer = layers.Embedding(max_words,embedding_dim)(text_input)
#embidding_text_flatten_layer = layers.Flatten()(embidding_text_layer)
LSTM_text_layer = layers.LSTM(32)(embidding_text_layer)
dense_text_layer = layers.Dense(32, activation='relu')(LSTM_text_layer)

#--------------- concatenated-----------------
concatenated = layers.concatenate([dense_digit_layer,LSTM_text_layer], axis=-1)

conc_layrs = layers.Dense(32, activation='relu')(concatenated)
conc_layrs = layers.Dense(32, activation='relu')(conc_layrs)
answer = layers.Dense(1)(conc_layrs)

model = Model([digit_input,text_input],answer)

#предобучение embidding векторами glowe

#model.layers[4].set_weights([embedding_matrix])
#model.layers[4].trainable = False

model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

history = model.fit([X_train, X_text_train], Y_train, epochs=5, batch_size=512,validation_data=([X_val, X_text_val], Y_val))
model.save_weights('pre_trained_glove_digit_text_api_model.h5')

#графики изменения качества модели

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc = history.history['val_acc']
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