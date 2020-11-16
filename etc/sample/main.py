import numpy as np
import tensorflow
from keras import Sequential, optimizers
from keras.layers import Dense

TEST_DATA_TIME = np.array([1, 2, 3, 4, 5, 6])
TEST_DATA_SCORE = np.array([10, 20, 30, 40, 50, 60])


def data():
    W = tensorflow.Variable(tensorflow.ones(shaple = (2, 2)), name = 'W')
    b = tensorflow.Variable(tensorflow.zeros(shaple = (2,)), name = 'b')

    @tensorflow.function
    def forward(x):
        return W * x + b

    out_a = forward([1, 0])
    print(out_a)


if __name__ == '__main__':
    # 모델 구성
    model = Sequential()
    model.add(Dense(1, input_dim = 1, activation = 'linear'))

    # 모델 컴파일
    sgd = optimizers.SGD(lr = 0.01)
    model.compile(optimizer = sgd, loss = 'mse', metrics = ['accuracy'])

    # 모델 학습
    model.fit(TEST_DATA_TIME, TEST_DATA_SCORE, batch_size = 1, epochs = 100, shuffle = False)

    # model.evaluate(verbose = 2)
    # 모델 예측
    print(model.predict([7]))
