import numpy as np
from keras import models, layers
from keras.datasets import imdb
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)


def vectorize_sequences(sequence, dimension = 10000):
    results = np.zeros((len(sequence), dimension))

    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1.

    return results


x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(units = 16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(units = 16, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x = x_test, y = y_test, epochs = 20, batch_size = 512)

history = history.history
loss = history['loss']
accuracy = history['accuracy']

plt.plot(range(1, len(loss) + 1), loss, 'r', label = 'Loss')
plt.plot(range(1, len(accuracy) + 1), accuracy, 'b', label = 'Accuracy')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

model.fit(x_train, y_train, epochs = 4, batch_size = 512)
result = model.evaluate(x_test, y_test)
