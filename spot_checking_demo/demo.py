import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
# 对数据缩放，若不缩放，蝴蝶结逻辑回归算法将会报错
scaler = MinMaxScaler()
X_rescaled = scaler.fit_transform(X)
kfold = KFold(n_splits=10)
# 逻辑回归算法默认的 solver 为 lbfgs，不缩放数据将会报错，若指定 solver 为 liblinear 则不需要再缩放数据
# model = LogisticRegression(solver='liblinear')
# 逻辑回归算法 KFold 评分结果：mean(平均值):0.68 var(方差):0.02 std(标准差):0.12
model = LogisticRegression()
# 随机数森林算法 KFold 评分结果：mean(平均值):0.66 var(方差):0.01 std(标准差):0.12
# model = RandomForestClassifier()
cv_score = cross_val_score(model, X_rescaled, y, cv=kfold)
# 打印平均值 方差 标准差
# 先看平均值，平均值越高说明算法更合适。若两个算法的平均值一样，再比较两个算法的标准差，标准差越小的算法说明更合适
print('mean(平均值):%.2f var(方差):%.2f std(标准差):%.2f' %(cv_score.mean(), cv_score.var(), cv_score.std()))
