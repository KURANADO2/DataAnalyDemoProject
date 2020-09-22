from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from sklearn import model_selection
from sklearn import metrics


class PredictDiabetes(object):
    def load_data(self):
        data_set = pd.read_csv('./pima-indians-diabetes.csv')
        feature_columns = ['pregnant',
                           'glucose',
                           'bp',
                           'skin',
                           'insulin',
                           'bmi',
                           'pedigree',
                           'age']
        return data_set, feature_columns

    def predict_diabetes(self, data_set, feature_columns):
        X_train = data_set[feature_columns]
        y_train = data_set['label']
        print(X_train, y_train)
        # 默认使用 gini 算法，可以使用指定算法，比较各个算法计算结果的准确度决定最终使用哪个算法，这个比较的过程叫做 spot check
        # 使用 entropy 信息增益算法
        dtc = DecisionTreeClassifier()
        # 开始训练
        dtc.fit(X_train, y_train)
        # 测试数据
        X_test = np.array([1,85,66,29,0,26.6,0.351,31]).reshape(1, -1)
        # 预测
        y_test = dtc.predict(X_test)
        print(y_test)
        dot_data = StringIO()
        # 将算法分析的每一步写到 out_file 变量中
        export_graphviz(dtc, out_file=dot_data,
                        feature_names=feature_columns,
                        class_names=['0', '1'],
                        filled=True,
                        rounded=True,
                        special_characters=True,
                        node_ids=True,
                        proportion=True,
                        rotate=True)
        # 生成 Graph 图
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # 生成 PDF 文件
        graph.write_pdf('decision_tree.pdf')

    def calculate_accuracy(self, data_set, feature_columns):
        X = data_set[feature_columns]
        y = data_set['label']
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        # 使用 entropy 信息增益算法，准确率达到了 0.7 - 0.8（使用默认的 gini 只能达到 0.5 - 0.6 左右）
        dtc = DecisionTreeClassifier(criterion='entropy')
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        print(metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    predict = PredictDiabetes()
    data_set, feature_columns = predict.load_data()
    predict.predict_diabetes(data_set, feature_columns)
    predict.calculate_accuracy(data_set, feature_columns)