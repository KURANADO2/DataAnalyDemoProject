import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import cluster
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import model_selection
import numpy as np


class EmployeeAna(object):
    # 加载数据
    def load_data(self):
        # 读取数据
        data_set = pd.read_csv('./employee_data.csv')
        # 数据预处理
        le = preprocessing.LabelEncoder()
        data_set['salary'] = le.fit_transform(data_set['salary'])
        data_set['Departments'] = le.fit_transform(data_set['Departments'])
        # print(data_set['salary'])
        print(data_set['Departments'])

        return data_set

    # 一次生成一张图表
    def stat_graph_generate(self, data_set):
        # 按照项目数分组统计
        number_projects = data_set.groupby('number_project').count()
        print(number_projects)
        plt.bar(x=number_projects.index.values, height=number_projects['satisfaction_level'])
        plt.xlabel('The number of projects')
        plt.ylabel('The number of employees')
        plt.show()
        # 按照在公司的工作年限分组统计
        time_spend_company = data_set.groupby('time_spend_company').count()
        print(time_spend_company)
        plt.bar(x=time_spend_company.index.values, height=time_spend_company['satisfaction_level'])
        plt.xlabel('The number of time speed')
        plt.ylabel('The number of employees')
        plt.show()

    # 一次生成多张图表
    def multi_stat_graph_generate(self, data_set):
        # 准备生成 10 个图表，尽量不要生成太多，否则会有很多内容会被遮挡，建议每张图片上最多保存 4 张图表，并以 2 行 2 列显示
        features = ['satisfaction_level',
                    'last_evaluation',
                    'number_project',
                    'average_montly_hours',
                    'time_spend_company',
                    'Work_accident',
                    'left',
                    'promotion_last_5years',
                    'Departments',
                    'salary'
                    ]
        for i, j in enumerate(features):
            # 多张图表以 5 行 3 列展示
            # plt.subplot(5, 3, i + 1)
            # 多张图表以 5 行 3 列展示
            plt.subplot(10, 1, i + 1)
            seaborn.countplot(x=j, data=data_set)
            # 调整水平间距，防止上一行中图表的 x 轴文字被下一行图表覆盖
            plt.subplots_adjust(hspace=1.0)
            # x 轴坐标文字太长会产生相互覆盖，可让文字旋转 90 度
            plt.xticks(rotation=90)
        plt.show()

    # 分析员工离职原因
    def ana_emp_left(self, data_set):
        print(data_set)
        # 分析员工离职与员工对公司的满意度和公司对员工的评价之间的关系
        left_emp = data_set[['satisfaction_level', 'last_evaluation']][data_set.left == 1]
        # 将员工分为 3 类分析
        kmeans = cluster.KMeans(n_clusters=3).fit(left_emp)
        left_emp['label'] = kmeans.labels_
        # 点状图
        plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'])
        plt.xlabel('satisfaction_level')
        plt.ylabel('last_evaluation')
        plt.show()

    # 预测哪些员工将要离职
    def predict_emp_left(self, data_set):
        features = ['satisfaction_level',
                    'last_evaluation',
                    'number_project',
                    'average_montly_hours',
                    'time_spend_company',
                    'Work_accident',
                    'promotion_last_5years',
                    'Departments',
                    'salary'
                    ]
        X = data_set[features]
        y = data_set['left']
        # 梯度递增算法
        gbc = GradientBoostingClassifier()
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        gbc.fit(X_train, y_train)
        y_pred = gbc.predict(X_test)
        # 预测准确度
        print(metrics.accuracy_score(y_test, y_pred))
        # 测试数据
        test = np.array([0.8, 0.86, 5, 263, 6, 0, 0, 7, 2]).reshape(1, -1)
        # 预测结果
        pred = gbc.predict(test)
        print(pred)


if __name__ == '__main__':
    ana = EmployeeAna()
    data_set = ana.load_data()
    # ana.stat_graph_generate(data_set)
    # ana.multi_stat_graph_generate(data_set)
    # ana.ana_emp_left(data_set)
    ana.predict_emp_left(data_set)
