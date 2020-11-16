import numpy as np


def vectorize_sequence(sequence, dimension = 10000):
    result = np.zeros((len(sequence), dimension))
    for i, seq in enumerate(sequence):
        result[i, seq] = 1.

    return result
