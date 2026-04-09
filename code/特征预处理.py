"""
案例: 岩石特征与处理之 归一化操作

回归: 特征工程的目的和步骤
    目的:
        利用专业的背景知识 和 技巧处理数据, 用于提升模型的性能
    步骤:
        1.特征提取
        2.特征预处理
        3.特征降维
        4.特征选择
        5.特征组合

    特征预处理之 归一化介绍
        目的:
            防止因为量纲问题, 导致特征列的方差相差较大, 影响模型的最终结果
        公式:
            x' = (当前值 - 该列最小值) / (该列最大值 - 该列最小值)
            x'' = x' * (mx - mi) + mi
        弊端:
            容易受到最大最小值的影响, 一般用于处理小数据集

    特征预处理之 标准化介绍
        目的:
            防止因为量纲问题, 导致特征列的方差相差较大, 影响模型的最终结果
            通过公式把 各列的值映射到均值为 0 , 标准查未 1 的正态分布序列
        公式:
            x' = (当前值 - 该列平均值) / 该列的标准差
        应用场景:
            适用于大数据集的处理
"""
#导包
from sklearn.preprocessing import MinMaxScaler #归一化对象

#准备数据集
x_train = [[90,2,10,40],[60,4,15,45],[75,3,13,46]]

#创建归一化对象
#参数 feature_range 表示生成范围, 默认为[0,1] 如果就是这个区间, 则参数可以省略
scaler = MinMaxScaler(feature_range=(3,5))

#对原数据进行归一化操作
x_train_scaled = scaler.fit_transform(x_train)

#打印数据
print(x_train_scaled)

#导包
from sklearn.preprocessing import StandardScaler

#创建标准化对象
scaler = StandardScaler()

#对原始数据进行标准化操作
x_train_scaled = scaler.fit_transform(x_train)

#打印标准化后的数据集
print(x_train_scaled)
