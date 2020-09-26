from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# data_set = pd.read_excel('data.xls', sheet_name='svm')
# print(data_set)
data_set = pd.read_csv('letter-recognition.csv')
feature_columns = data_set.columns[1:]
X = data_set[feature_columns]
y = data_set.letter
# print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
