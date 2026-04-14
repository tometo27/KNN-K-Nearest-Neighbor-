"""
案例:
    演示逻辑回归模型实现 癌症预测

逻辑回归模型介绍
    概述
        属于有监督学习
        主要适用于: 二分类
    原理:
        把线性回归处理后的预测值 -> 通过 sigmoid 激活函数, 映射到[0,1]概率 -> 基于自定义的阈值,结合概率分类
    损失函数:
        极大似然估计函数的负数形式
    机器学习基本流程
        加载数据
        数据处理
        特征工程
        模型训练
        模型预测
        模型评估
"""
#导包
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression #逻辑回归模型
from sklearn.preprocessing import StandardScaler #标准化
from sklearn.model_selection import train_test_split #数据划分
from sklearn.metrics import accuracy_score #模型评估
from streamlit import dataframe
# 1. 加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()
# 2. 数据预处理
# 要替换的数据,替换成什么?,是否更改原数据
data.replace('?',np.nan,inplace=True)
# 缺失值处理
data.dropna(inplace=True)
# data.info()
# 3.特征工程
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
print(x[:5])
print(y[:5])
print(x.shape,y.shape)
#划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#创建标准化对象
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
#模型训练
#创建模型对象
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
#模型预测
y_pred = estimator.predict(x_test)
print(y_pred)
#模型评估
#预测前评估, 利用x_test,y_test进行评估
print(estimator.score(x_test, y_test))
#预测后评估, 利用y_test 和 t_pred评估
print(accuracy_score(y_test, y_pred))
#注意逻辑回归可以使用准确率来评估, 但是评估不准确