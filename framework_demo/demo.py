# 选出最优算法
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def load_data_set():
    return make_classification(n_samples=1000, n_classes=2)


def define_models():
    models = dict()
    models['LR'] = LogisticRegression()
    models['KNN'] = KNeighborsClassifier()
    models['GNB'] = GaussianNB()
    models['DCT'] = DecisionTreeClassifier()
    models['SVC'] = SVC()
    return models


def make_pipeline(model):
    steps = list()
    steps.append(('Standardize', StandardScaler()))
    steps.append(('Normalize', Normalizer()))
    steps.append(('model', model))
    return Pipeline(steps)


def evaluate_single_model(X, y, model, folds):
    pipeline = make_pipeline(model)
    cv_score = cross_val_score(pipeline, X, y, cv=folds, scoring='accuracy')
    return cv_score


def evaluate_models(X, y, models, folds=10):
    results = dict()
    skf = StratifiedKFold(n_splits=folds)
    for name, model in models.items():
        cv_score = evaluate_single_model(X, y, model, skf)
        if cv_score is not None:
            results[name] = cv_score
            print('name:%s mean(平均值):%.2f var(方差):%.2f std(标准差):%.2f' % (
            name, cv_score.mean(), cv_score.var(), cv_score.std()))
        else:
            print('%s model error', name)
    return results


def show_graph(results):
    plt.boxplot(x=results.values(), labels=results.keys())
    plt.show()


if __name__=='__main__':
    X, y = load_data_set()
    # print(X, y)
    models = define_models()
    results = evaluate_models(X, y, models)
    show_graph(results)
