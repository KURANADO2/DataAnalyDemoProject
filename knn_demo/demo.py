from sklearn import datasets
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

# data_sets = datasets.load_digits()
# # 数字 0 feature
# print(data_sets.data[0])
# # target
# print(data_sets.target)
# # 数字 3 的训练图片
# plt.matshow(data_sets.images[3])
# plt.show()


class DigitIdentify(object):
    def digit_identify(self):
        X, y = datasets.load_digits(return_X_y=True)
        # data_set = datasets.load_digits()
        knn = neighbors.KNeighborsClassifier()
        knn.fit(X, y)
        # knn.fit(data_set.data, data_set.target)
        test_data = [0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 9, 10, 0, 15, 4, 0,
                     0, 0, 3, 16, 12, 14, 2, 0,
                     0, 0, 4, 16, 16, 2, 0, 0,
                     0, 3, 16, 8, 10, 13, 2, 0,
                     0, 1, 15, 1, 3, 16, 8, 0,
                     0, 0, 11, 16, 15, 11, 1, 0]
        test_result = knn.predict(np.array(test_data).reshape(1, -1))
        print(test_result)


if __name__ == '__main__':
    di = DigitIdentify()
    di.digit_identify()
