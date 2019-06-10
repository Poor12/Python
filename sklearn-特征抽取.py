from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,KernelPCA
from sklearn import datasets
digits=datasets.load_digits()
features=StandardScaler().fit_transform(digits.data)
pca=PCA(n_components=0.99,whiten=1)#n_components是方差
features_pca=pca.fit_transform(features)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_pca.shape[1])

#线性不可分
features,_=datasets.make_circles(n_samples=1000,random_state=1,noise=0.1,factor=0.1)
print(features)
kpca=KernelPCA(kernel="rbf",gamma=15,n_components=1)#特征
features_kpca=kpca.fit_transform(features)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_kpca.shape[1])

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris=datasets.load_iris()
features=iris.data
target=iris.target
lda=LinearDiscriminantAnalysis(n_components=1)
features_lda=lda.fit(features,target).transform(features)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_lda.shape[1])
print(lda.explained_variance_ratio_)

#验证取特征数
lda=LinearDiscriminantAnalysis(n_components=None)
features_lda=lda.fit(features,target)
lda_var_ratios=lda.explained_variance_ratio_
print(lda_var_ratios)
def select_n_components(var_ratios,goal_var:float)->int:
    total_var=0.0
    n_components=0
    for explained_var in var_ratios:
        total_var+=explained_var
        n_components+=1
        if total_var>=goal_var:
            break
    return n_components
print(select_n_components(lda_var_ratios,0.95))

#NMF non-negative matrix factorization
from sklearn.decomposition import NMF
digits=datasets.load_digits()
features=digits.data
nmf=NMF(n_components=10,random_state=1)
features_nmf=nmf.fit_transform(features)
print("Original number of features:",features.shape[1])
print("Reduced number of features:",features_nmf.shape[1])

#TSVD 稀疏特征
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
digits=datasets.load_digits()
features=StandardScaler().fit_transform(digits.data)
features_sparse=csr_matrix(features)
print(features_sparse)
tsvd=TruncatedSVD(n_components=10)
features_sparse_tsvd=tsvd.fit_transform(features_sparse)
print("Original number of features:",features_sparse.shape[1])
print("Reduced number of features:",features_sparse_tsvd.shape[1])

