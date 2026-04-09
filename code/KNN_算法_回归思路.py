#导包
from sklearn.neighbors import KNeighborsRegressor
#准备数据集
x_train = [[0,0,1],[1,1,0],[3,3,10],[4,11,12]]
y_train = [0.1,0.2,0.3,0.4]
x_text = [[3,11,10]]

#创建模型对象
estimator = KNeighborsRegressor(n_neighbors=2)

#训练模型
estimator.fit(x_train, y_train)

#模型预测
y_pre = estimator.predict(x_text)
print(y_pre)