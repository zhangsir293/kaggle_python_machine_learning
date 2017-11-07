# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as pit
import numpy as np
#����sklearn�е��߼�˹�ٻع������
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv(r'D:\machine_learning\Datasets\Breast-Cancer\breast-cancer-train.csv')
df_test = pd.read_csv(r'D:\machine_learning\Datasets\Breast-Cancer\breast-cancer-test.csv')

df_test_nagetive=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive=df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]

#scatter Ϊ����������ɢ��ͼ,ԭ�㣬��ɫ
pit.scatter(df_test_nagetive['Clump Thickness'],
            df_test_nagetive['Cell Size'],marker='o',s=200,c='red')

pit.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'],marker='x',s=150,c='black')

#����x,y���˵��
pit.xlabel('Clump Thickness')
pit.ylabel('Cell Size')
pit.show()

#�����ʼ�������ķ�����
#����numpy�е�random�����������ֱ�ߵĽؾ�intercept��ϵ��coef
#random.random([x,y])����һ��x*y�����������
intercept = np.random.random([1])
coef = np.random.random([2])
"""
����һ��0-11������
һ��ֱ��
ax+by+c=0
"""
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]
#����һ�����ֱ��
pit.plot(lx,ly,c='yellow')

#ʹ��sklearn�е��߼�˹�ٻع������
lr=LogisticRegression()
#ʹ��ǰʮ��ѵ������ѧϰֱ�ߵ�ϵ���ͽؾ�
#fit(X, y, sample_weight=None)
"""
score(X,Y,sample_weight)
X������Y��ǩ,sample_weightȨ�أ�Ĭ�϶���1�����ز������ݵ�ƽ����ȷ��
fit(self, X, y, sample_weight=None):
����ѵ������߼�˹��ģ��(self)
"""
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],
                df_train['Type'][:10])
print('Testing accuracy(10 training samples):',
      lr.score(df_test[['Clump Thickness','Cell Size']],
               df_test['Type']))
intercept = lr.intercept_
coef = lr.coef_[0,:]
#��άƽ��
ly = (-intercept-lx*coef[0])/coef[1]

"""
����ʹ��ǰʮ������ѵ���õ����߼��ع�ͼ��
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
ʹ������ѵ������ѧϰֱ�ߵ�ϵ���ͽؾ�
"""
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print('Testing accuracy(all training samples):',
      lr.score(df_test[['Clump Thickness','Cell Size']],
               df_test['Type']))
"""
������������ѵ���õ����߼��ع�ͼ��
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

