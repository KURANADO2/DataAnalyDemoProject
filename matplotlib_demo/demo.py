import matplotlib.pyplot as plt


# plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.show()

value = [0.1, 0.2, 0.4, 0.3]
label = ['A', 'B', 'C', 'D']
# Python 统计分析中 x 参数一般都用来表示数据
# autopct 指定保留几位小数
plt.pie(x=value, labels=label, autopct='%.1f%%')
plt.show()

# 关于 matplotlib 更多图表使用方法，请参考官方示例：https://matplotlib.org/gallery/index.html