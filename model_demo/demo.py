from sklearn.datasets import load_digits
# 最近临近算法
from sklearn.neighbors import KNeighborsClassifier
# 决策树算法
from sklearn.tree import DecisionTreeClassifier
# 朴素贝叶斯算法
from sklearn.naive_bayes import GaussianNB
# Support Vector Machine 支持向量机算法
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
print(X, y)
models = [('KNN', KNeighborsClassifier()), ('DTC', DecisionTreeClassifier()), ('GNB', GaussianNB()), ('SVM', SVC())]
names = []
result_set = []
for name, model in models:
    skf = StratifiedKFold(n_splits=10)
    # 加上 scoring = 'accuracy' 参数，计算结果将更加精确（不加也可以）
    cv_score = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    names.append(name)
    result_set.append(cv_score)
    print('name:%s mean(平均值):%.2f var(方差):%.2f std(标准差):%.2f' % (name, cv_score.mean(), cv_score.var(), cv_score.std()))
    print(cv_score)
# 股市也是用 box 图
plt.boxplot(result_set, labels=names)
# 显示图表，橙色线表示中位线，中位线越靠上，算法越好，每个算法上下两条横线分别表示 score 的最大值和最小值，最小值越小算法越差
plt.show()
