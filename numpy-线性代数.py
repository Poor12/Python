#���Դ���
a=np.mat("0 1 2;1 0 3;4 -3 8")
print(np.linalg.inv(a))
#������Է�����
#b=np.mat([0,8,-9]).T
b=np.array([0,8,-9])
print(b)
x=np.linalg.solve(a,b)
print("solution",x)
print("check",np.dot(a,x))

#����ֵ����������
a=np.mat("3 -2;1 0")
print("eigenvalues",np.linalg.eigvals(a))
eigenvalues,eigenvectors=np.linalg.eig(a)
print("first",eigenvalues,'\n',"second",eigenvectors)
u,sigma,v=np.linalg.svd(a,full_matrices=0)
print(u,sigma,v)
print(u*np.diag(sigma)*v)

#���������invֻ���ܷ�����Ϊ���룬pinv��û���������
a=np.mat("4 11 14;8 7 -2")
pseudoinv=np.linalg.pinv(a)
print("pseudo inverse\n",pseudoinv)
print("check\n",a*pseudoinv)

#����ʽ
#������������ʽ
a=np.mat([[3,4],[5,6]])
print("determinant",np.linalg.det(a))

#���ٸ���Ҷ�任
#һ�ָ�Ч����ɢ����Ҷ�㷨��FFT�㷨�ȸ���ֱ�Ӽ������
x=np.linspace(0,2*np.pi,30)
wave=np.cos(x)
transformed=np.fft.fft(wave)#����Ҷ�任
print(np.all(np.abs(np.fft.ifft(transformed)-wave)<10**-9))#��ԭ�ź�
plt.plot(transformed)
plt.show()

#��Ƶ
x=np.linspace(0,2*np.pi,30)
wave=np.cos(x)
transformed=np.fft.fft(wave)
shifted=np.fft.fftshift(transformed)#��ֱ�������ƶ���Ƶ�׵��м�
print(np.all(np.fft.ifftshift(shifted)-transformed)<10**-9)#��ԭ�ź�
plt.plot(transformed,lw=2)
plt.plot(shifted,lw=3)
plt.show()

#�����
#����ֲ�
cash=np.zeros(10000)
cash[0]=1000
outcome=np.random.binomial(9,0.5,size=len(cash))# �׾Ŵ�Ӳ�ң�p=0.5,10000����������
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

#�����ηֲ�--��������������Ʒ���޷ŻصĴ��г�ȡָ������������󣬳��ָ���������������
points=np.zeros(100)
outcomes=np.random.hypergeometric(25,1,3,size=len(points))#��һ������Ϊ������ͨ����������ڶ�������Ϊ��ù�������������������Ϊÿ�β���������
for i in range(len(points)):
    if outcomes[i]==3:#��ͨ������
        points[i]=points[i-1]+1
    elif outcomes[i]==2:
        points[i]=points[i-1]-6
    else:
        print(outcomes[i])
plt.plot(np.arange(len(points)),points)
plt.show()

#�����ֲ�
#��̬�ֲ�
#������̬�ֲ�
