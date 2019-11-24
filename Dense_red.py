#%%
import numpy as np

# загружаем данные с фичами

dataset = np.loadtxt("DataSet/winequality-red.txt", delimiter=";")

X = dataset[:, 0:-1]
X = np.asarray(X).astype('float32')
Y = dataset[:, -1:]
Y = np.asarray(Y).astype('int')


def to_one_hot(labels, demension):
    results = np.zeros((len(labels), demension))
    step_shift = labels.min()
    for i, label in enumerate(labels):
        results[i, int(label - step_shift)] = 1
    return results


size_output_demension = int(Y.max() - Y.min() + 1)
Y = to_one_hot(Y, size_output_demension)

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
from keras import regularizers
#%%
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(size_output_demension, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, epochs=34, batch_size=32, validation_split=0.3)
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
