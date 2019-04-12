#����
A=np.mat('1 2 3;4 5 6;7 8 9')
B=np.matrix('1 2 3;4 5 6;7 8 9',copy=0)
C=np.mat(np.arange(9).reshape(3,3))
print(A,B)
print(A.T) #ת��
print(A.I)#�����
print(np.bmat("A C;A C"))

def ultimate_answer(a):
    result=np.zeros_like(a)#����һ����a��״��ͬ��ֵ��Ϊ0������
    result.flat=42
    return result
ufunc=np.frompyfunc(ultimate_answer,1,1)#ָ���������Ϊ1���������Ϊ1
print("the answer",ufunc(np.arange(4)))
print("the answer",ufunc(np.arange(4).reshape(2,2)))

#ͨ�ú������ĸ�������ֻ�������������������һ��������ufunc������Ч
a=np.arange(4)#0 1 2 3
b=np.arange(4)
print(np.add(a,b))
print(np.add.reduce(a))
print(np.add.accumulate(a))#�����м���
print(np.add.reduceat(a,[0,3,1]))#0-3��Լ����3 3-1��Լ����a[3] 1-:��Լ����6
print(np.add.outer(np.arange(2),a))
print(np.mat('1 2 3;4 5 6').A)#����תarray

#���������+��-��*��ʽ������ͨ�ú���add��subtract��multiply��
#���������Ĺ������Ϊ���ӣ�������ĳ����������漰����ͨ�ú���divide��true_divide��floor_division���Լ�������Ӧ�������/��//
a=np.array([2,6,5])
b=np.array([1,2,3])
print("divide",np.divide(a,b),np.divide(b,a))
print("true divide",np.true_divide(a,b),np.true_divide(b,a))#���ظ��������
print("floor divide",np.floor_divide(a,b),np.floor_divide(b,a))#��������ȡ��
print("/ divide",a/b,b/a)
print("// divide",a//b,b//a)

#ģ����
a=np.arange(-4,4)
print("% operator",np.remainder(a,2),np.mod(a,2),a%2)#����ȡ���ڳ���
print("fmod",np.fmod(a,2))#����ȡ���ڱ�����

#��������������
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import sys
a,b=9,8
t=np.linspace(-np.pi,np.pi,201)
x=np.sin(a*t+np.pi/2)
y=np.sin(b*t)
plot(x,y)
show()

#��ת������λ
x=np.arange(-9,9)
y=-x
print("sign difference?",(x^y)<0)
print("sign difference?",np.less(np.bitwise_xor(x,y),0))
print("power of 2?\n",x,"\n",(x&(x-1))==0)
print("power of 2?\n",x,"\n",np.equal(np.bitwise_and(x,x-1),0))
#���������ļ���ʵ����ֻ����ģΪ2������ʱ��Ч
#�����κ�ͬ�ŵ�������������mod��ʹ�̾�����С��������ŵ�����������C/javaʹ�̾����ܴ�python��ʹ�̾�����С
print("modules 4\n",x,"\n",x&(1<<2-1))
print("modules 4\n",x,"\n",np.bitwise_and(x,np.left_shift(1,2)-1))