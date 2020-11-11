import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# 텐서는 데이터를 위한 컨테이너 입니다.
# 거의 항상 수치형 데이터를 다루므로 숫자를 위한 컨테이너 입니다.


# [스칼라] 0차원 텐서 - 하나의 숫자를 가짐
do = np.array(1)

# [벡터] 1차원 텐서 - 하나의 축을 가짐
d1 = np.array(range(1, 11))

# [행렬] 2차원 텐서 - 2개의 축을 가짐
d2 = np.array([
    range(1, 11),
    range(11, 21)
])

# 3차원 텐서
d3 = np.array([
    [
        range(1, 11),
        range(11, 21),
    ],
    [
        range(31, 41),
        range(41, 51),
    ]
])

(train_image, train_label), (test_image, test_label) = mnist.load_data()

digit = train_image[0]
plt.imshow(digit)
plt.show()
