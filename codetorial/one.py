# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
from tensorflow import keras

"""
http://codetorial.net/tensorflow/index.html

1. MNIST 데이터 셋 임포트
load_data() 를 통해 4개의 변수에 NumPy Array 를 반환
x_train, x_test 는 28x28 픽셀의 손글씨 이미지 데이터
y_train, y_test 는 분류를 위한 0~9 레이블 값
"""
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
2. 데이터 전처리
0~255.0 값을 갖는 픽셀값들을 0~1.0으로 변환
"""
x_train, x_test = x_train / 255.0, x_test / 255.0

if __name__ == '__main__':
    """
    3. 모델 구성 
    Flatten() 으로 28x28 픽셀의 값을 784개의 1차원 배열로 변환
    다음 두개의 뉴런 층은 Dense() 로 Fully-connected layer 를 구성
    각 층은 512개와 10개의 인공 뉴런 노드를 갖고 activation function 으로 ReLU, softmax 를 사용
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = (28, 28)),
        keras.layers.Dense(512, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    """
    4. 모델 컴파일
    학습 과정에서 Loss function 를 줄이기 위해(?)1 Optimizer 로 Adam(Adaptive Momentum estimation) 를 사용하며,
    Loss function 은 sparse_categorical_crossentrophy 를 지정하고 평가지표로 accuracy 를 사용.
    """
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    """ 5. 모델 훈련 """
    # model.fit(x_train, y_train, epochs = 5)

    """ 6. 정확도 평가 """
    # test_loss, test_acc = model.evaluate(x_test, y_test)

    loss, accuracy = [], []
    for i in range(10):
        model.fit(x_train, y_train, epochs = 1)
        ev_loss, ev_acc = model.evaluate(x_test, y_test)
        loss.append(ev_loss)
        accuracy.append(ev_acc)

    print(loss)
    print(accuracy)
    """
    실행 Console log 
    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 13s 219us/sample - loss: 0.2005 - accuracy: 0.9414
    Epoch 2/5
    60000/60000 [==============================] - 11s 183us/sample - loss: 0.0808 - accuracy: 0.9752
    Epoch 3/5
    60000/60000 [==============================] - 11s 183us/sample - loss: 0.0521 - accuracy: 0.9838
    Epoch 4/5
    60000/60000 [==============================] - 10s 166us/sample - loss: 0.0369 - accuracy: 0.9881
    Epoch 5/5
    60000/60000 [==============================] - 11s 183us/sample - loss: 0.0276 - accuracy: 0.9912

    10000/10000 [==============================] - 1s 76us/sample - loss: 0.0659 - accuracy: 0.9810
    """
