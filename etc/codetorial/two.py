import numpy
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

    weights = model.get_weights()
    # print(weights, len(weights))

    """
    첫번째 어레이는 784x512개의 값을 갖는 2차원 어레이로서 입력층 (input layer)과 은닉층 (hidden layer)을 연결하는 가중치를 나타내는 값입니다.
    두번째 어레이는 512개의 0으로 이루어져 있으며, 은닉층 (hidden layer)의 바이어스 (bias) 값을 나타냅니다.
    세번째 어레이는 512x10개의 값을 갖는 2차원 어레이로서 은닉층 (hidden layer)과 출력층 (output layer)을 연결하는 가중치를 나타내는 값입니다.
    네번째 어레이는 10개의 0으로 이루어져 있으며, 출력층 (output layer)의 바이어스 (bias) 값을 나타냅니다.
    (784, 512)
    (512,)
    (512, 10)
    (10,)
    """
    print(weights[0].shape)
    print(weights[1].shape)
    print(weights[2].shape)
    print(weights[3].shape)

    """ 결과 csv 파일 저장 """
    # numpy.savetxt('weights[0].csv', weights[0])
    # numpy.savetxt('weights[0].shape.csv', weights[0].shape)

    weight0 = numpy.random.rand(784, 512) * 0.1
    weight1 = numpy.zeros(512)
    weight2 = numpy.random.rand(512, 10) * 0.05
    weight3 = numpy.zeros(10)
    weights2 = numpy.array([weight0, weight1, weight2, weight3])

    print(weights2[0].shape)
    print(weights2[1].shape)
    print(weights2[2].shape)
    print(weights2[3].shape)
