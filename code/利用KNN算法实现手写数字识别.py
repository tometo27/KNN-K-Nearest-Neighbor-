import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

from sympy.integrals.rationaltools import ratint_logpart


# 定义函数, 接收用户传入的索引, 展示该索引对应的图片
def show_digit(idx):
    df = pd.read_csv('D:\develop\PyCharm\code\my_project\data\手写数字识别.csv')
    #判断传入的索引是否越界
    if idx< 0 or idx>= len(df) - 1:
        print('error')
        return
    # 获取数据
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    print(f'{y.iloc[idx]}')

    # 查看x的形状
    print(x.iloc[idx].shape)
    print(x.iloc[idx].values)
    x = x.iloc[idx].values.reshape(28,28)
    print(x)


    #展示灰度图
    plt.imshow(x, cmap='gray')
    plt.show()

# 定义函数, 训练模型, 并保存训练好的模型'
def train_model():
    df = pd.read_csv('D:\develop\PyCharm\code\my_project\data\手写数字识别.csv')
    # 拆分特征列
    x = df.iloc[:,1:]
    # 拆分标签列
    y = df.iloc[:,0]
    # 打印特征和标签的形状
    print(x.shape)
    print(y.shape)
    # 对特征列进行归一化
    x = x/255
    #拆分训练集和测试集
    #参1特征列 参2 标签列 参3 测试集比例 参4 随机种子 参5 参考y值进行抽取, 保持标签的比例
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=47, stratify=y)
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 模型训练
    estimator.fit(x_train,y_train)
    #模型评估
    print(estimator.score(x_test,y_test))
    print(accuracy_score(y_test,estimator.predict(x_test)))
    #保存模型
    joblib.dump(estimator, './my_model/手写数字识别.pkl')

def test_model():
    #加载图片
    x = plt.imread('./data/demo.png')
    #绘制图片
    # plt.imshow(x, cmap='gray')
    # plt.show()
    # print(x)
    #加载模型
    estimator = joblib.load('./my_model/手写数字识别.pkl')
    #使用模型
    x = x.reshape(1,-1)
    y_pre = estimator.predict(x)
    print(y_pre)


if __name__ == '__main__':
   test_model()
