import os
os.environ['PYTHONHASHSEED'] = str(0)

#%%
import numpy as np

np.random.seed(2)

# загружаем данные с фичами

dataset = np.loadtxt("/home/krisantis/Desktop/Vine/red/winequality-red.txt",
                     delimiter=";")

X = dataset[:, 0:-2]
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

from keras import models
from keras import layers
#%%
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=14, batch_size=32, validation_split=0.3)
#model.save_weights('Dense_model.h5')

#графики изменения качества модели
#%%
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

# %%
