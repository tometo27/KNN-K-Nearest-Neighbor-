from sklearn.datasets import load_iris #加载鸢尾花数据集
import  seaborn as sns
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集
from sklearn.preprocessing import StandardScaler  #数据标准化
from sklearn.neighbors import KNeighborsClassifier  #KNN算法, 分类对象
from sklearn.metrics import accuracy_score  #模型评估,计算模型预测的准确率

"""
案例: 演示网格搜索 和 交叉验证
交叉验证解释:
    原理:
        第 1 次：用折 2345 训练，折 1 测试
        第 2 次：用折 1345 训练，折 2 测试
        第 3 次：用折 1245 训练，折 3 测试
        第 4 次：用折 1235 训练，折 4 测试
        第 5 次：用折 1234 训练，折 5 测试
        计算上述五次准确率的平均值, 作为模型最终的准确率
        
        假设第4次最好, 则用全部数据训练模型, 再次用测试集对模型测试
    目的:
        为了让模型的最终验证结果更加准确
网格搜索:
    目的:
        寻找最优超参数
    原理:
        接受超参可能出现的值,然后针对于超惨的每个值进行交叉验证, 获取到最优超参组合
    超参数:
        需要用户手动录入的数据,不同的超参,可能会影响模型的最终评测结果
"""

#加载鸢尾花数据集
iris = load_iris()

#数据预处理, 这里是切分训练集和测试集比例 8 : 2
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=47)

#创建标准化对象
transfer = StandardScaler()

x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#模型训练
#创建KNN分类对象
estimator = KNeighborsClassifier()
#定义字典 记录超参出现的情况
param_dict = {'n_neighbors':[i for i in range(1,11)],}
#参1 要计算最优超参的模型对象, 参2 该模型超参可能出现的值, 参3 交叉验证的折数
estimator = GridSearchCV(estimator,param_dict,cv =5)
estimator.fit(x_train,y_train)
#打印最优超参组合
print(estimator.best_score_)
print(estimator.best_params_)
print(estimator.best_estimator_)
print(estimator.cv_results_)

#获取最优超参的模型对象
estimator = KNeighborsClassifier(n_neighbors=11)
#模型训练
estimator.fit(x_train,y_train)
#模型预测
y_pre = estimator.predict(x_test)
#模型评估
print(accuracy_score(y_test,y_pre))