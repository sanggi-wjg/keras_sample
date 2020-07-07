import autokeras as ak

from keras import models
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils

# 1. 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 2. 모델 구성
model = models.Sequential()
model.add(Dense(activation = 'relu', input_dim = 28 * 28, output_dim = 64, ))
model.add(Dense(activation = 'relu', output_dim = 10))

# 3. 모델 컴파일
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 4. 모델 학습
model.fit(x_train, y_train, epochs = 5, batch_size = 32)

# 5. 모델 사용하기
loss_metrics = model.evaluate(x_test, y_test, batch_size = 32)

# clf = ak.ImageClassifier()
# clf.fit(x_train, y_train)
# results = clf.predict(x_test)
