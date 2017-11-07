# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as pit
import numpy as np
#导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv(r'D:\machine_learning\Datasets\Breast-Cancer\breast-cancer-train.csv')
df_test = pd.read_csv(r'D:\machine_learning\Datasets\Breast-Cancer\breast-cancer-test.csv')

df_test_nagetive=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive=df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]

#scatter 为良性肿瘤做散点图,原点，红色
pit.scatter(df_test_nagetive['Clump Thickness'],
            df_test_nagetive['Cell Size'],marker='o',s=200,c='red')

pit.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'],marker='x',s=150,c='black')

#绘制x,y轴的说明
pit.xlabel('Clump Thickness')
pit.ylabel('Cell Size')
pit.show()

#随机初始化参数的分类器
#利用numpy中的random函数随机采样直线的截距intercept和系数coef
#random.random([x,y])生成一个x*y的随机数矩阵
intercept = np.random.random([1])
coef = np.random.random([2])
"""
生成一个0-11的数组
一条直线
ax+by+c=0
"""
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]
#绘制一条随机直线
pit.plot(lx,ly,c='yellow')

#使用sklearn中的逻辑斯蒂回归分类器
lr=LogisticRegression()
#使用前十条训练样本学习直线的系数和截距
#fit(X, y, sample_weight=None)
"""
score(X,Y,sample_weight)
X样本，Y标签,sample_weight权重，默认都是1，返回测试数据的平均精确度
fit(self, X, y, sample_weight=None):
返回训练后的逻辑斯蒂模型(self)
"""
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],
                df_train['Type'][:10])
print('Testing accuracy(10 training samples):',
      lr.score(df_test[['Clump Thickness','Cell Size']],
               df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0,:]
#二维平面
ly = (-intercept-lx*coef[0])/coef[1]

"""
绘制使用前十个数据训练得到的逻辑回归图像
"""
pit.plot(lx,ly,c='green')
pit.scatter(df_test_nagetive['Clump Thickness'],
            df_test_nagetive['Cell Size'],marker='o',s=200,c='red')
pit.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'],marker='x',s=150,c='black')
pit.xlabel('Clump Thickness')
pit.ylabel('Cell Size')
pit.show()
#lr=LogisticRegression()
"""
使用所有训练样本学习直线的系数和截距
"""
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print('Testing accuracy(all training samples):',
      lr.score(df_test[['Clump Thickness','Cell Size']],
               df_test['Type']))
"""
绘制所有数据训练得到的逻辑回归图像
"""
intercept = lr.intercept_
coef = lr.coef_[0,:]
ly = (-intercept-lx*coef[0])/coef[1]

pit.plot(lx,ly,c='red')
pit.scatter(df_test_nagetive['Clump Thickness'],
            df_test_nagetive['Cell Size'],marker='o',s=200,c='green')
pit.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'],marker='x',s=150,c='blue')
pit.xlabel('Clump Thickness')
pit.ylabel('Cell Size')
pit.show()

