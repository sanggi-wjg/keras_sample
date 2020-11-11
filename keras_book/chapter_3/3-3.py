import numpy as np

from keras import models, layers
from keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

# 평균을 빼준 후 표준편차로 나누어 줌으로 데이터 정규화
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model


K = 4

num_val_samples = len(train_data) // K
num_epochs = 100
all_scores = []

for i in range(K):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_target = train_target[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]
    ], axis = 0)
    partial_train_target = np.concatenate([
        train_target[:i * num_val_samples], train_target[(i + 1) * num_val_samples:]
    ], axis = 0)

    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs = num_epochs, batch_size = 1, verbose = 0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose = 0)
    all_scores.append(val_mae)
