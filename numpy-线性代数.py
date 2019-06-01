#线性代数
a=np.mat("0 1 2;1 0 3;4 -3 8")
print(np.linalg.inv(a))
#求解线性方程组
#b=np.mat([0,8,-9]).T
b=np.array([0,8,-9])
print(b)
x=np.linalg.solve(a,b)
print("solution",x)
print("check",np.dot(a,x))

#特征值和特征向量
a=np.mat("3 -2;1 0")
#矩阵的秩
#非0的奇异值=矩阵的秩
print("eigenvalues",np.linalg.eigvals(a))
eigenvalues,eigenvectors=np.linalg.eig(a)
print("first",eigenvalues,'\n',"second",eigenvectors)
u,sigma,v=np.linalg.svd(a,full_matrices=0)
print(u,sigma,v)
print(u*np.diag(sigma)*v)

#广义逆矩阵，inv只接受方阵作为输入，pinv则没有这个限制
a=np.mat("4 11 14;8 7 -2")
pseudoinv=np.linalg.pinv(a)
print("pseudo inverse\n",pseudoinv)
print("check\n",a*pseudoinv)

#行列式
#计算矩阵的行列式
a=np.mat([[3,4],[5,6]])
print("determinant",np.linalg.det(a))

#快速傅里叶变换
#一种高效的离散傅里叶算法，FFT算法比根据直接计算更快
x=np.linspace(0,2*np.pi,30)
wave=np.cos(x)
transformed=np.fft.fft(wave)#傅里叶变换
print(np.all(np.abs(np.fft.ifft(transformed)-wave)<10**-9))#还原信号
plt.plot(transformed)
plt.show()

#移频
x=np.linspace(0,2*np.pi,30)
wave=np.cos(x)
transformed=np.fft.fft(wave)
shifted=np.fft.fftshift(transformed)#将直流分量移动到频谱的中间
print(np.all(np.fft.ifftshift(shifted)-transformed)<10**-9)#还原信号
plt.plot(transformed,lw=2)
plt.plot(shifted,lw=3)
plt.show()

#随机数
#二项分布
cash=np.zeros(10000)
cash[0]=1000
outcome=np.random.binomial(9,0.5,size=len(cash))# 抛九次硬币，p=0.5,10000次上述过程
for i in range(1,len(cash)):
    if outcome[i]<5:
        cash[i]=cash[i-1]-1
    elif outcome[i]<10:
        cash[i]=cash[i-1]+1
    else:
        raise AssertionError("unexcepted outcome"+outcome)
print(outcome.min(),outcome.max())
plt.plot(np.arange(len(cash)),cash)
plt.show()

#超几何分布--罐子里有两种物品，无放回的从中抽取指定数量的物件后，抽出指定种类物件的数量
points=np.zeros(100)
outcomes=np.random.hypergeometric(25,1,3,size=len(points))#第一个参数为罐中普通球的数量，第二个参数为倒霉球的数量，第三个参数为每次采样的数量
for i in range(len(points)):
    if outcomes[i]==3:#普通球数量
        points[i]=points[i-1]+1
    elif outcomes[i]==2:
        points[i]=points[i-1]-6
    else:
        print(outcomes[i])
plt.plot(np.arange(len(points)),points)
plt.show()

#连续分布
#正态分布
#对数正态分布
