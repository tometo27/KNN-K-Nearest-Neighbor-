"""
案例:
    演示 欠拟合, 正好拟合, 过拟合, L1正则化, L2正则化的效果图

    过拟合解决办法: 手动筛选特征, L1和L2正则化
    欠拟合解决办法: 增加特征, 从而增加模型的复杂度

    L1和L2正则化
        目的/思路:
            都是基于惩罚系数来修改权重, 乘法系数越大, 则修改力度越大,对应权重就越小
        区别
            L1正则化可以实现权重变为 0 , 从而达到特征选择的目的
            L2正则化, 只能让权重无限趋近于0 , 但是不能为 0

"""
# 导包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge,Lasso

#定义函数模拟欠拟合
def dm01_under_fitting():
    #指定随机种子, 固定每次生成的数据
    np.random.seed(42)
    #随机生成x轴100个数据模拟特征
    x = np.random.uniform(-3,3,100)
    #基于x轴, 通过线性公式生成 y 轴 100个数据
    y = 0.5* x ** 2 + np.random.normal(0,1,100)
    #数据预处理
    #把特征转化成一列多行的形式
    X = x.reshape(-1,1)
    #创建模型对象
    estimator = LinearRegression()
    estimator.fit(X, y)
    #模型预测
    y_pred = estimator.predict(X)
    #模型评估
    print(mean_squared_error(y,y_pred))
    #绘图
    plt.scatter(X,y)
    plt.plot(X,y_pred,color='red')
    plt.show()

def dm02_just_fitting():
    # 指定随机种子, 固定每次生成的数据
    np.random.seed(42)
    # 随机生成x轴100个数据模拟特征
    x = np.random.uniform(-3, 3, 100)
    # 基于x轴, 通过线性公式生成 y 轴 100个数据
    y = 0.5 * x ** 2 + np.random.normal(0, 1, 100)
    # 数据预处理
    # 把特征转化成一列多行的形式
    X = x.reshape(-1, 1)
    #拼接一个数组,形成正好拟合
    X1 = np.hstack([X, X ** 2])
    # 创建模型对象
    estimator = LinearRegression()
    estimator.fit(X1, y)
    # 模型预测
    y_pred = estimator.predict(X1)
    # 模型评估
    print(mean_squared_error(y, y_pred))
    # 绘图
    plt.scatter(X, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red') #基于x做排序,并返回排序后的索引
    plt.show()

def dm03_upper_fitting():
    # 指定随机种子, 固定每次生成的数据
    np.random.seed(42)
    # 随机生成x轴100个数据模拟特征
    x = np.random.uniform(-3, 3, 100)
    # 基于x轴, 通过线性公式生成 y 轴 100个数据
    y = 0.5 * x ** 2 + np.random.normal(0, 1, 100)
    # 数据预处理
    # 把特征转化成一列多行的形式
    X = x.reshape(-1, 1)
    # 拼接一个数组,形成正好拟合
    X1 = np.hstack([X, X ** 2,X ** 3,X ** 4,X ** 5,X ** 6,X ** 7,X ** 8,X ** 9,X ** 10])
    # 创建模型对象
    estimator = LinearRegression()
    estimator.fit(X1, y)
    # 模型预测
    y_pred = estimator.predict(X1)
    # 模型评估
    print(mean_squared_error(y, y_pred))
    # 绘图
    plt.scatter(X, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')  # 基于x做排序,并返回排序后的索引
    plt.show()

#定义函数模拟L1正则化
def dm04_L1():
    # 指定随机种子, 固定每次生成的数据
    np.random.seed(42)
    # 随机生成x轴100个数据模拟特征
    x = np.random.uniform(-3, 3, 100)
    # 基于x轴, 通过线性公式生成 y 轴 100个数据
    y = 0.5 * x ** 2 + np.random.normal(0, 1, 100)
    # 数据预处理
    # 把特征转化成一列多行的形式
    X = x.reshape(-1, 1)
    # 拼接一个数组,形成正好拟合
    X1 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    # 创建模型对象
    # estimator = LinearRegression()
    #改为创建L1正则化对象
    estimator = Lasso(alpha=0.1)
    estimator.fit(X1, y)
    # 模型预测
    y_pred = estimator.predict(X1)
    # 模型评估
    print(mean_squared_error(y, y_pred))
    plt.scatter(X, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()
def dm05_L2():
    # 指定随机种子, 固定每次生成的数据
    np.random.seed(42)
    # 随机生成x轴100个数据模拟特征
    x = np.random.uniform(-3, 3, 100)
    # 基于x轴, 通过线性公式生成 y 轴 100个数据
    y = 0.5 * x ** 2 + np.random.normal(0, 1, 100)
    # 数据预处理
    # 把特征转化成一列多行的形式
    X = x.reshape(-1, 1)
    # 拼接一个数组,形成正好拟合
    X1 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])
    # 创建模型对象
    # estimator = LinearRegression()
    #改为创建L2正则化对象
    estimator = Ridge(alpha=0.1)
    estimator.fit(X1, y)
    # 模型预测
    y_pred = estimator.predict(X1)
    # 模型评估
    print(mean_squared_error(y, y_pred))
    plt.scatter(X, y)
    plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red')
    plt.show()

if __name__ == '__main__':
    # dm01_under_fitting()
    # dm02_just_fitting()
    # dm03_upper_fitting()
    dm05_L2()