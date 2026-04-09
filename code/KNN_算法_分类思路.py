"""
KNN 算法介绍
    原理:
        基于欧式距离计算 测试集 和 每个训练集之间的距离根据距离升序排列找到最近的K个样本
        基于K个样本投票, 票数最多的作为最终预测结果 -> 分类问题
        基于K个样本计算平均值, 作为最终预测结果 -> 回归问题
    实现思路:
        1.分类问题
            适用于: 有特征, 有标签, 且标签是不连续的
        2.回归问题
            适用于: 有特征, 有标签, 且标签连续
"""
# 导包
from sklearn.neighbors import KNeighborsClassifier
# 准备数据集
x_train = [[0],[1],[2],[3]]
y_train = [0,0,1,1]
x_text = [[5]]

#创建KNN模型对象
#estimator: 估计器, 模型对象 , 也可以用变量名 model 做接收
estimator = KNeighborsClassifier(n_neighbors=4)

#模型训练
#传入:训练集的特征数据, 训练集的标签数据
estimator.fit(x_train, y_train)

#模型预测
#传入: 测试集的特证数据, 获取列: 测试结果
y_pre = estimator.predict(x_text)

print(y_pre)
