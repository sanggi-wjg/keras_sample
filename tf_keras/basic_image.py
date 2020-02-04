import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def show_mnist_sample():
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_mnist_sample_2():
    plt.figure(figsize = (10, 10,))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])

    plt.show()


if __name__ == '__main__':
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape = (28, 28,)),
    #     keras.layers.Dense(128, activation = 'relu'),
    #     keras.layers.Dense(10, activation = 'softmax'),
    # ])

    # model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
    # model.fit(train_images, train_labels, epochs = 5)
    show_mnist_sample_2()
    # predictions = model.predict(test_images)
    # np.argmax(predictions[0])