Numpy Scientific computing http://numpy.org
Pandas data analysis and manipulation tool
Matplotlib generate graph http://matplotlib.org
sk-learn environment important

clustering 分组
jiangwei

数字存 1000 万训练集
字母存 2000 万训练集

概率论
线性代数
离散数学

KNN K-Nearest Neighbor 最邻近算法

数据集分为训练集和测试集
训练集数据越多，精确度越高
原则上测试集不应该超过 1/4，若测试集超过 50%，则没有什么意义
测试集用于测试训练结果的准确度
一般 data(feature) 为 X，target(label) 为 y
X_train 训练集数据
y_train 训练集 target
X_test 测试集数据
y_test 测试集 target

n-fold

逻辑回归算法

决策树分类算法
用例：根据体检各项指标，使用决策树判定得到语料建议 - 曲阳医院

graphviz 以图像化方式展示决策树
pip install pydotplus
对于不是 .whl 而是 .tar.gz 的依赖，解压后执行 python setup.py install 即可安装

使用预处理（preprocessing）对数据预处理，将原有数据中的单词等数据转换成数学代号，如数字，以方便分析

梯度递增算法