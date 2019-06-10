#特征选择
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
#Variance thresholding
iris=datasets.load_iris()
features=iris.data
target=iris.target
threholder=VarianceThreshold(threshold=.5)
features_high_variance=threholder.fit_transform(features)
print(features_high_variance[0:3])

#create feature matrix with：
#Feature 0:0.8 class 0
#Feature 1:0.8 class 1
#Feature 2:0.6 class 0,0.4 class 1
#E(X1)=0.2 E(X2)=3.2,E(X3)=0.8
features=[[0,1,0],
          [0,1,1],
          [0,1,0],
          [0,1,1],
          [1,0,0]]
threholder=VarianceThreshold(threshold=.75*(1-.75))
print(threholder.fit_transform(features))

#处理高关联的特征
features=np.array([[1,1,1],
                   [2,2,0],
                   [3,3,1],
                   [4,4,0],
                   [5,5,1],
                   [6,6,0],
                   [7,7,1],
                   [8,7,0],
                   [9,7,1]])
dataframe=pd.DataFrame(features)
#关联矩阵
corr_matrix=dataframe.corr().abs()
#选择上三角
upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
#找到关联大于0.95的特征列
to_drop=[column for column in upper.columns if any(upper[column]>0.95)]
#丢掉特征
print(dataframe.drop(dataframe.columns[to_drop],axis=1).head(3))

#特征是绝对的,卡方告诉我们两个特征的相关性
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif
iris=load_iris()
features=iris.data
target=iris.target
#将数据转换成int
features=features.astype(int)
#选择两个卡方统计较高的特征
chi2_selector=SelectKBest(chi2,k=2)
features_kbest=chi2_selector.fit_transform(features,target)
#展示结果
print("Original number of features:",features.shape[1])
print("reduced number of features:",features_kbest.shape[1])

#特征是数量上的
fvalue_selector=SelectKBest(f_classif,k=2)
features_kbest=fvalue_selector.fit_transform(features,target)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_kbest.shape[1])

fvalue_selector=SelectPercentile(f_classif,percentile=75)
features_kbest=fvalue_selector.fit_transform(features,target)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_kbest.shape[1])

#递归消除特征
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets,linear_model

warnings.filterwarnings(action="ignore",module="scipy",message="^internal gelsd")
features,target=make_regression(n_samples=10000,n_features=100,n_informative=2,random_state=1)
print(features.shape,target)
#创建一个线性回归
ols=linear_model.LinearRegression()
#递归消除特征
rfecv=RFECV(estimator=ols,step=1,scoring="neg_mean_squared_error")
print(rfecv.fit(features,target).transform(features))
print(rfecv.n_features_)
print(rfecv.support_)
print(rfecv.ranking_.shape)