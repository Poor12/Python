import numpy as np
from sklearn import preprocessing
feature=np.array([[-500.5],
                  [-100.1],
                  [0],
                  [100.1],
                  [900.9]])
minmax_scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
scaled_feature=minmax_scaler.fit_transform(feature)
print(scaled_feature)

#标准化
sd_scaler=preprocessing.StandardScaler()
sd_feature=sd_scaler.fit_transform(feature)
print(sd_feature)

#每个数除以L2范数
features=np.array([[0.5,0.5]])
normalizer=preprocessing.Normalizer(norm='l2')
print(normalizer.transform(features))

features=np.array([[2,3],
                   [2,3],
                   [2,3]])
polynomial_interaction=preprocessing.PolynomialFeatures(degree=2,include_bias=False)
poly_feature=polynomial_interaction.fit_transform(features)
print(poly_feature)

#转换特征
def add_ten(x):
    return x+10
ten_transformer=preprocessing.FunctionTransformer(add_ten)
ten_result=ten_transformer.fit_transform(features)
print(ten_result)

import pandas as pd
df=pd.DataFrame(features,columns=['features_1','features_2'])
print(df.apply(add_ten))

#处理异常点
houses=pd.DataFrame()
houses['Price']=[534433,392333,293222,4322032]
houses['Bathrooms']=[2,3.5,2,116]
houses['Square_Feet']=[1500,2500,1500,48000]
houses_re=houses[houses['Bathrooms']<20]
houses["outlier"]=np.where(houses["Bathrooms"]<20,0,1)
print(houses)

age=np.array([[6],
              [12],
              [20],
              [36],
              [65]])
binarier=preprocessing.Binarizer(18)
print(binarier.fit_transform(age))
#multi threholds
print(np.digitize(age,bins=[20,30,64]))

#kmeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
features,_=make_blobs(n_samples=50,n_features=2,centers=3,random_state=1)
dataframe=pd.DataFrame(features,columns=["feature_1","feature_2"])
clusterer=KMeans(3,random_state=0)
clusterer.fit(features)
dataframe['group']=clusterer.predict(features)
print(dataframe.head(5))

#删除缺失值
features=np.array([[1.1,11.1],
                   [2.2,22.2],
                   [3.3,33.3],
                   [np.nan,44.4]])
print(features[~np.isnan(features).any(axis=1)])
dataframe=pd.DataFrame(features,columns=["feature_1","feature_2"])
print(dataframe.dropna())
print(dataframe[~dataframe["feature_1"].isnull()])

#插入缺失值
features,_=make_blobs(n_samples=1000,n_features=2,random_state=1)
scaler=preprocessing.StandardScaler()
sd_features=scaler.fit_transform(features)
true_value=sd_features[0,0]
sd_features[0,0]=np.nan
mean_imputer=preprocessing.Imputer(strategy='mean',axis=0)
features_mean_imputed=mean_imputer.fit_transform(features)
print("true value:",true_value)
print("imputed value:",features_mean_imputed[0,0])

#one-hot
feature=np.array([["Texas"],
                  ["California"],
                  ["Texas"],
                  ["Delaware"]])
one_hot=preprocessing.LabelBinarizer()
one_hot_re=one_hot.fit_transform(feature)
print(one_hot_re)
print(one_hot.classes_)
print(one_hot.inverse_transform(one_hot.transform(feature)))
print(pd.get_dummies(feature[:,0]))
multiclass_feature=[("Texas","Florida"),
                    ("California","Alabama"),
                    ("Texas","Alabama")]
one_hot_multi=preprocessing.MultiLabelBinarizer()
print(one_hot_multi.fit_transform(multiclass_feature))

dataframe=pd.DataFrame({"Score":["Low","Low","Medium","Medium","High"]})
scaler_mapper={"Low":1,"Medium":2,"High":3}
print(dataframe["Score"].replace(scaler_mapper))

#字典特征
from sklearn.feature_extraction import DictVectorizer
data_dict=[{"Red":2,"Blue":4},
           {"Red":4,"Blue":3}]
dictvectorizer=DictVectorizer(sparse=False)
features=dictvectorizer.fit_transform(data_dict)
print(features)
feature_names=dictvectorizer.get_feature_names()
print(pd.DataFrame(features,columns=feature_names))

#KNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
X=np.array([[0,2.10,1.45],
            [1,1.18,1.33],
            [0,1.22,1.27],
            [1,-0.21,-1.19]])
X_nan=np.array([[np.nan,0.87,1.31]])
clf=KNeighborsClassifier(3,weights='distance')
trained_model=clf.fit(X[:,1:],X[:,0])
imputed_values=trained_model.predict(X_nan[:,1:])
X_imputed=np.hstack((imputed_values.reshape(-1,1),X_nan[:,1:]))
X_re=np.vstack((X_imputed,X))
print(X_re)

