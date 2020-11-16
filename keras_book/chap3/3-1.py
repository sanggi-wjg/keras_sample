import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers, optimizers
from keras.datasets import imdb
from keras.layers import Dense

from keras_book.chap3 import vectorize_sequence

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)

# train_data 0번째 인덱스 문장으로 출력
# word_index = imdb.get_word_index()
# word_index = dict((value, key) for (key, value) in word_index.items())
#
# for index in train_data[0]:
#     print(word_index.get(index - 3), end = ' ')


# 신경망에 숫자 리스트를 주입할 수 없으니, 텐서로 변경
x_train, y_train = vectorize_sequence(train_data), np.asarray(train_label).astype('float32')
x_test, y_test = vectorize_sequence(test_data), np.asarray(test_label).astype('float32')

model = models.Sequential()
model.add(Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = optimizers.RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train[:10000], y_train[:10000], epochs = 20, batch_size = 512, validation_data = (x_train[10000:], y_train[10000:]))
result_history = history.history
val_loss, val_acc, loss, acc = result_history['val_loss'], result_history['val_accuracy'], result_history['loss'], result_history['accuracy']

epochs = range(1, len(val_loss) + 1)
plt.rcParams["figure.figsize"] = (20, 15)
plt.rcParams["axes.grid"] = True

plt.subplot(211)
plt.plot(epochs, loss, 'bo', label = 'Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.legend()

plt.subplot(212)
plt.plot(epochs, acc, 'bo', label = 'Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.legend()

plt.show()

model.evaluate(x_test, y_test, batch_size = 512)
