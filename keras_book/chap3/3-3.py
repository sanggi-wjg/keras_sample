import numpy as np

from keras import models
from keras.datasets import boston_housing
from keras.layers import Dense

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data -= mean
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(Dense(64, activation = 'relu', input_shape = (train_data.shape[1],)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model


K, EPOCHS = 4, 100
NUM_SAMPLES = len(train_data) // K
all_score = []

for i in range(K):
    val_data = train_data[i * NUM_SAMPLES: (i + 1) * NUM_SAMPLES]
    val_target = train_target[i * NUM_SAMPLES: (i + 1) * NUM_SAMPLES]

    partial_train_data = np.concatenate([
        train_data[:i * NUM_SAMPLES], train_data[(i + 1) * NUM_SAMPLES:]
    ], axis = 0)
    partial_train_target = np.concatenate([
        train_target[:i * NUM_SAMPLES], train_target[(i + 1) * NUM_SAMPLES:]
    ], axis = 0)

    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs = EPOCHS, batch_size = 1, verbose = 0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose = 0)
    all_score.append(val_mae)
