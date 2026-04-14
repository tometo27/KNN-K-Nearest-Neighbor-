"""
案例:
    演示混淆矩阵 和 精确率, 召回率, F1值
逻辑回归:
    有监督学习,有特征有标签 且 标签分离
    适用于二分类标记方式

    评估:
        精确率, 召回率, F1值

混淆矩阵:
    作用:
        用来描述 真实值 和 预测值之间的关系
    图解:
                    预测标签( 正例 )        预测标签( 反例 )
        真实标签        真正例                 伪反例
        预测标签        伪正例                 真反例
    单词
    Positive :正例
    Negative :反例
    结论
        模拟使用分类少的充当正例
        精确率 = 真正例 在 预测正例中的占比
        召回率 = 真正例 在 真正例中的占比
        F1值 = 2*( 精确率*召回率 ) / ( 精确率 *召回率 )
"""
#导包
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

#定义变量 记录: 样本数据
y_train = ['恶性','恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']
#定义变量记录模型a的预测结果
y_pred_A = ['恶性','恶性','恶性','恶性','良性','良性','良性','良性','良性','良性','良性']
#定义变量记录模型b的预测结果
y_pred_B = ['恶性','恶性','恶性','恶性','恶性','恶性','恶性','良性','恶性','恶性','良性']
cm_A = confusion_matrix(y_train, y_pred_A)
label = ['恶性','良性']
df_label = ['恶性(正例)','良性(反例)']

cm_A = confusion_matrix(y_train, y_pred_A,labels=label)
df_A = pd.DataFrame(cm_A,index=df_label,columns=df_label)
print(df_A)
cm_B = confusion_matrix(y_train, y_pred_B,labels=label)
df_B = pd.DataFrame(cm_B,index=df_label,columns=df_label)
print(df_B)
#计算A模型的精确率,召回率,F1值
print(f'精确率:{precision_score(y_train, y_pred_A,pos_label='恶性')}'
      f'\n召回率:{recall_score(y_train,y_pred_A,pos_label='恶性')}'
      f'\nf1:{f1_score(y_train,y_pred_A,pos_label='恶性')}')
