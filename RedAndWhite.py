#%%
import numpy as np

# загружаем данные с фичами
datasetRed = np.loadtxt("DataSet/winequality-red.txt", delimiter=";")
datasetWhite = np.loadtxt("DataSet/winequality-white.txt", delimiter=";")

#формируем едины массив красного и белое вина
X = np.concatenate((datasetRed[:, 0:-1], datasetWhite[:, 0:-1]))
Y = np.concatenate((datasetRed[:, -1:], datasetWhite[:, -1:]))

#приведение типов
#метки в int т.к. многоклассовая однозначная классификация
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('int')


#векторизация меток(целей)
#количество меток рассчитывается исходя из датасета
def to_one_hot(labels, demension):
    results = np.zeros((len(labels), demension))
    step_shift = labels.min()
    for i, label in enumerate(labels):
        results[i, int(label - step_shift)] = 1
    return results


size_output_demension = int(Y.max() - Y.min() + 1)
Y = to_one_hot(Y, size_output_demension)

print("DataSet:\nattributes: ", X.shape[1], "\nsamples:", X.shape[0],
      "\nclass labels:", size_output_demension)

#Перемешивание вариантов
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

#разделение датасета на тренировочный и тестовый
validation_split = 0.3
validation_samples = int(X.shape[0] * validation_split)
train_samples = X.shape[0] - validation_samples

X_train = X[:train_samples]
Y_train = Y[:train_samples]
X_val = X[train_samples:]
Y_val = Y[train_samples:]

print("train_samples: ", train_samples, "\nvalidation_samples:",
      validation_samples)

#нормализация по тренировочной выборке
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std

X_val -= mean
X_val /= std

#импорт библиотек Keras
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import RMSprop

#построение модели
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dropout(0.15))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(size_output_demension, activation='softmax'))

#обучение модели
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    Y_train,
                    epochs=100,
                    batch_size=64,
                    validation_data=[X_val, Y_val])

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