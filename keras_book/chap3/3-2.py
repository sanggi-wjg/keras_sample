from keras import models
from keras.datasets import reuters
from keras.layers import Dense
from keras.utils import to_categorical

from keras_book.chap3 import vectorize_sequence

(train_data, train_label), (test_data, test_label) = reuters.load_data(num_words = 10000)

x_train, x_label = vectorize_sequence(train_data), vectorize_sequence(train_label, 46)
y_train, y_label = vectorize_sequence(test_data), vectorize_sequence(test_label, 46)

x_label_categorical = to_categorical(train_label)
y_label_categorical = to_categorical(test_label)

model = models.Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (10000,)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
history = model.fit(x_train[1000:], x_label_categorical[1000:], epochs = 20, batch_size = 512, validation_data = (x_train[:1000], x_label_categorical[:1000]))
