#矩阵
A=np.mat('1 2 3;4 5 6;7 8 9')
B=np.matrix('1 2 3;4 5 6;7 8 9',copy=0)
C=np.mat(np.arange(9).reshape(3,3))
print(A,B)
print(A.T) #转置
print(A.I)#逆矩阵
print(np.bmat("A C;A C"))

def ultimate_answer(a):
    result=np.zeros_like(a)#创建一个与a形状相同，值都为0的数组
    result.flat=42
    return result
ufunc=np.frompyfunc(ultimate_answer,1,1)#指定输入参数为1，输出参数为1
print("the answer",ufunc(np.arange(4)))
print("the answer",ufunc(np.arange(4).reshape(2,2)))

#通用函数有四个方法，只对输入两个参数、输出一个参数的ufunc对象有效
a=np.arange(4)#0 1 2 3
b=np.arange(4)
print(np.add(a,b))
print(np.add.reduce(a))
print(np.add.accumulate(a))#保存中间结果
print(np.add.reduceat(a,[0,3,1]))#0-3规约返回3 3-1规约返回a[3] 1-:规约返回6
print(np.add.outer(np.arange(2),a))
print(np.mat('1 2 3;4 5 6').A)#矩阵转array

#基本运算符+、-、*隐式关联着通用函数add、subtract和multiply。
#除法包含的过程则较为复杂，在数组的除法运算中涉及三个通用函数divide、true_divide和floor_division，以及两个对应的运算符/和//
a=np.array([2,6,5])
b=np.array([1,2,3])
print("divide",np.divide(a,b),np.divide(b,a))
print("true divide",np.true_divide(a,b),np.true_divide(b,a))#返回浮点数结果
print("floor divide",np.floor_divide(a,b),np.floor_divide(b,a))#返回向上取整
print("/ divide",a/b,b/a)
print("// divide",a//b,b//a)

#模运算
a=np.arange(-4,4)
print("% operator",np.remainder(a,2),np.mod(a,2),a%2)#正负取决于除数
print("fmod",np.fmod(a,2))#正负取决于被除数

#绘制莉萨如曲线
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import sys
a,b=9,8
t=np.linspace(-np.pi,np.pi,201)
x=np.sin(a*t+np.pi/2)
y=np.sin(b*t)
plot(x,y)
show()

#玩转二进制位
x=np.arange(-9,9)
y=-x
print("sign difference?",(x^y)<0)
print("sign difference?",np.less(np.bitwise_xor(x,y),0))
print("power of 2?\n",x,"\n",(x&(x-1))==0)
print("power of 2?\n",x,"\n",np.equal(np.bitwise_and(x,x-1),0))
#计算余数的技巧实际上只用于模为2的幂数时有效
#对于任何同号的两个整数，其mod都使商尽可能小；对于异号的两个整数，C/java使商尽可能大，python等使商尽可能小
print("modules 4\n",x,"\n",x&(1<<2-1))
print("modules 4\n",x,"\n",np.bitwise_and(x,np.left_shift(1,2)-1))