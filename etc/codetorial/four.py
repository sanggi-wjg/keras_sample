from tensorflow import keras
from tensorflow.keras.utils import plot_model

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if __name__ == '__main__':
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    plot_model(model, to_file = 'model.png')
    plot_model(model, to_file = 'model_shape.png', show_shapes = True)

# OSError: `pydot` failed to call GraphViz.Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH.
