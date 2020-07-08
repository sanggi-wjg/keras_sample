import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers
from keras.datasets import reuters
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 1000)


# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decode_word = [reverse_word_index.get(i - 3, '?') for i in train_data[0]]

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, s in enumerate(sequences):
        results[i, s] = 1.
    return results


x_train = vectorize_sequences(train_data)
# y_train = np.asarray(train_labels).astype('float32')
y_train = to_categorical(train_labels)

x_test = vectorize_sequences(test_data)
# y_test = np.asarray(test_labels).astype('float32')
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(units = 64, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(units = 64, activation = 'relu'))
model.add(layers.Dense(units = 46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_data = (x_train[:1000], y_train[:1000]))
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']
history_acc = history.history['accuracy']
history_val_acc = history.history['val_accuracy']

epochs = range(1, len(history_loss) + 1)
plt.plot(epochs, history_loss, 'bo', label = 'Training Loss')
plt.plot(epochs, history_val_loss, 'b', label = 'Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, history_acc, 'bo', label = 'Training Acc')
plt.plot(epochs, history_val_acc, 'b', label = 'Validation Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

result = model.evaluate(x_test, y_test)
