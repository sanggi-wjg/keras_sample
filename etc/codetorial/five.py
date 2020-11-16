# import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

if __name__ == '__main__':
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    history = model.fit(x_train, y_train, validation_split = 0.25, epochs = 10, verbose = 1)
    print(history.history)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()
