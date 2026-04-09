from sklearn.datasets import load_iris #加载鸢尾花数据集
import  seaborn as sns
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  #分割训练集和测试集
from sklearn.preprocessing import StandardScaler  #数据标准化
from sklearn.neighbors import KNeighborsClassifier  #KNN算法, 分类对象
from sklearn.metrics import accuracy_score  #模型评估,计算模型预测的准确率

#1. 定义函数, 加载鸢尾花数据集, 并查看数据集
# def dm01_loadiris():
#     # 加载鸢尾花数据集
#     iris_data = load_iris()
#     # 查看数据集
#     print(iris_data)

# 定义函数, 绘制数据集的散点图
def dm02_showIris():
    iris_data = load_iris()
    # 把iris封装成df对象
    iris_df = pd.DataFrame(data=iris_data.data, columns = iris_data.feature_names)
    # 给df新增一列充当标签列
    iris_df['label'] = iris_data.target
    # 通过seaborn 绘制散点图
    # 参1 数据集 , 参2 x , y轴, 标签 参3 是否生成拟合回归线
    sns.lmplot(data = iris_df, x = 'sepal length (cm)' , y = 'sepal width (cm)', hue = 'label', fit_reg = True)
    plt.title('iris data')
    plt.tight_layout()#自动调整子图参数,使图像便捷与子图匹配
    plt.show()

#数据集划分
def dm03_split_train_test():
    iris_data = load_iris()
    # 数据预处理 从 150个特征和标签 中按照 8:2的比例切分训练集和测试集
    # 随机种子用于对数据'洗牌',打乱测试数据


# 模型训练和预测
def dm04_iris_evaluate_test():
    iris_data = load_iris()

    # 数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,random_state=47)
    print(y_test)
    # 特征工程
    # 创建标准化对象
    transfer = StandardScaler()

    # 对特征列进行标准化
    #fit_transform 适用于第一次进行标准化的时候使用, 先训练在转换, 一般用于训练集
    #transfor 只转换, 该函数适用于重复进行标准化时使用, 一般用于对测试集进行标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)

    #具体的训练模型的操作
    estimator.fit(x_train, y_train)

    #模型评估和预测
    #模型预测
    #场景一 对刚才切分的30条进行测试
    y_pre = estimator.predict(x_test)
    print(y_pre)

    #模型评估
    # 直接评分 基于训练集的特征和标签
    print(estimator.score(x_test, y_test))
    #基于测试集的特征和标签评分
    print(accuracy_score(y_test, y_pre))



if __name__ == '__main__':
    dm04_iris_evaluate_test()


