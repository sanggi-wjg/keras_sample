from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_image, train_label), (test_image, test_label) = mnist.load_data()
train_image_copy, train_label_copy, test_image_copy, test_label_copy = train_image, train_label, test_image, test_label

model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (28 * 28,)))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_image = train_image.reshape((60000, 28 * 28))
train_image = train_image.astype('float32') / 255
train_label = to_categorical(train_label)

test_image = test_image.reshape((10000, 28 * 28))
test_image = test_image.astype('float32') / 255
test_label = to_categorical(test_label)

model.fit(train_image, train_label, epochs = 5, batch_size = 128)
test_loss, test_acc = model.evaluate(test_image, test_label)
