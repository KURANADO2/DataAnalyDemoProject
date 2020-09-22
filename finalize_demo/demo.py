import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

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
model = LogisticRegression(solver='liblinear')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
file_name = 'finalize_model.ml'
# 固化模型，将模型保存到 16 进制文件中
pickle.dump(model, open(file_name, 'wb'))
# 加载模型（如果之前已经保存过模型，则只要拿到模型文件，就可以直接对测试数据进行分析，不用再去加载训练集开始训练、选择算法，直接就可以开始分析、预测）
loaded_model = pickle.load(open('finalize_model.ml', 'rb'))
y_pred = loaded_model.predict(X_test)
print(y_pred)
