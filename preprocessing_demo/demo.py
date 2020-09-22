import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
import numpy as np

data_set = pd.read_csv('./pima-indians-diabetes.csv')
feature_columns = ['pregnant',
                   'glucose',
                   'bp',
                   'skin',
                   'insulin',
                   'bmi',
                   'pedigree',
                   'age']
X = data_set[feature_columns]
y = data_set.label
# 最大最小值缩放
# scaler = MinMaxScaler()
# 标准缩放
# scaler = StandardScaler()
# 规范化
# scaler = Normalizer()
# 二进制化(转换成 0 和 1，输出时不需要再设置小数精度)
scaler = Binarizer()
# 数据缩放
X_rescaled = scaler.fit_transform(X)
# 设置精准度，保留 3 位有效小数
# np.set_printoptions(precision=3)
print(X_rescaled)
