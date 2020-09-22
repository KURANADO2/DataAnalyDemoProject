from sklearn import cluster
import matplotlib.pyplot as plt


# 加载数据并对数据进行预处理
def load_data(path):
    file = open(path, 'r')
    return file.readlines()


def pre_precess(lines):
    customer_names = []
    data_sets = []
    for i in range(len(lines)):
        line_arr = lines[i].strip().split(',')
        customer_names.append(line_arr[0])
        items = []
        # for j in range(1, len(line_arr)):
        #     items.append(float(line_arr[j]))
        # data_sets.append(items)
        # 上面三行代码可简写为下面一行
        data_sets.append([float(line_arr[j]) for j in range(1, len(line_arr))])
    return customer_names, data_sets


# 分组
def kmeans_cluster(customer_names, data_sets):
    kmeans = cluster.KMeans(n_clusters=3)
    # 分析和预测，得到分组结果
    labels = kmeans.fit_predict(X=data_sets)
    print('分组结果:', labels)
    # 存放分组后的用户名
    cluster_customer_names = [[], [], []]
    for i in range(len(custome_names)):
        cluster_customer_names[labels[i]].append(custome_names[i])
    return cluster_customer_names


def show_graph(cluster_customer_names):
    values = []
    for i in range(len(cluster_customer_names)):
        values.append(len(cluster_customer_names[i]))
    labels = ['first group', 'second group', 'third group']
    plt.pie(x=values, labels=labels)
    plt.show()


if __name__ == '__main__':
    lines = load_data('customers.txt')
    custome_names, data_sets = pre_precess(lines)
    print(custome_names)
    print(data_sets)
    cluster_customer_names = kmeans_cluster(custome_names, data_sets)
    for i in range(len(cluster_customer_names)):
        print(cluster_customer_names[i])
    show_graph(cluster_customer_names)
