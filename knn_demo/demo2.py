from sklearn import datasets
from sklearn import neighbors
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# KNN 算法
class DigitIdentify(object):
    def digit_identify_knn(self):
        X, y = datasets.load_digits(return_X_y=True)
        # KNN K-Nearest Neighbor 最邻近算法
        knn = neighbors.KNeighborsClassifier()
        # 划分训练集和测试集
        # 测试集用于测试训练结果的准确度
        # 训练集数据越多，精确度越高
        # 原则上测试集不应该超过 1 / 4，若测试集超过 50 %，则没有什么意义
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
        # 传入训练集的 X 和 y 开始训练
        knn.fit(X_train, y_train)
        # 传入测试集的 X 开始预测，得到预测结果
        y_pred = knn.predict(X_test)
        # 传入测试集真正的 y 和测试集预测出的 y，计算预测准确度
        print(metrics.accuracy_score(y_test, y_pred))

    # 决策树算法
    def digit_identify_dec_tree(self):
        X, y = datasets.load_digits(return_X_y=True)
        dec_tree = DecisionTreeClassifier()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
        dec_tree.fit(X_train, y_train)
        y_pred = dec_tree.predict(X_test)
        print(metrics.accuracy_score(y_test, y_pred))

    # 逻辑回归算法
    def digit_identify_lr(self):
        X, y = datasets.load_digits(return_X_y=True)
        # 默认的 solver 为 lbfgs，会报错误信息
        lr = LogisticRegression(solver='liblinear')
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print(metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    di = DigitIdentify()
    di.digit_identify_knn()
    di.digit_identify_dec_tree()
    di.digit_identify_lr()
