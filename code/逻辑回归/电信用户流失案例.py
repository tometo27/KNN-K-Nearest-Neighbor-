#导包
from narwhals import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#数据导入
df_cu = pd.read_csv('./data/churn.csv')
print(df_cu.head())
#因为Churn和gender字段是字符串, 所以需要进行one-hot编码, 热编码处理
df_cu = pd.get_dummies(df_cu, columns=['Churn','gender'])
#删除one-hot新增的冗余列
df_cu.drop(['Churn_No','gender_Male'],axis = 1,inplace = True)
#修改列名, 将Churn改为flag充当标签列
df_cu.rename(columns = {'Churn_Yes':'flag'},inplace = True)
print(df_cu.flag.count())
#数据可视化,hue分组
# sns.countplot(data = df_cu, x = 'Contract_Month',hue = 'flag')
# plt.show()
#提取特征列和标签列
x = df_cu[['Contract_Month','internet_other','PaymentBank']]
y = df_cu['flag']
#划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
#特征预处理(不需要)
#模型训练
#创建逻辑回归对象
estimator = LogisticRegression()
estimator.fit(x_train,y_train)
#模型预测
y_pred = estimator.predict(x_test)
print(y_pred)
#模型评估
print(estimator.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))
#精确率
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
#宏平均: 不考虑数据权重,直接求权重,适用于样本均衡的情况
#样本权重平均 考虑样本数据权重, 根据权重求加权平均, 适用于样本不均衡的情况
print(classification_report(y_test,y_pred))

#查看处理后的数据集
# df_cu.info()
