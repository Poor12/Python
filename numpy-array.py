from numpy import *
import sys
from datetime import *
#向量
#0-n的整数的平方和立方的表达
# def pythonsum(n):
#     a=range(n)
#     b=range(n)
#     c=[]
#     for i in range(len(a)):
#         c.append(i**2+i**3)
#     return c
#
# def numpysum(n):
#     a=arange(n)**2
#     b=arange(n)**3
#     c=a+b
#     return c
#
# size=4000
# start=datetime.now()
# c=pythonsum(size)
# delta=datetime.now()-start
# print(c[-2:])
# print("pythonsum elapsed time in the microseconds",delta.microseconds)
# start=datetime.now()
# d=numpysum(size)
# delta=datetime.now()-start
# print(d[-2:])
# print("numpysum elapsed time in the microseconds",delta.microseconds)

#行向量
vector_row=np.array([1,2,3])
#列向量
vector_column=np.array([[1],
                        [2],
                        [3]])

a=arange(5)
print(a.shape)
b=array([arange(2),arange(2)])
print(b.shape)
c=arange(4,7,dtype='f')
print(c)
#自定义数据类型
t=dtype([('name',str_,40),('numitems',int32),('price',float32)])
print(t)
itemz=array([('meaning of life dvd',42,3.14),('buffer',13,2.72)])
print(itemz[1,1])

#一维数组的索引与切片
print(a[:4:2])
print(a[::-1])

#多维数组
d=arange(24).reshape(2,3,4)
print(d)
print(d[:,0,0])
print(d[0,:,:],'\n',d[0,...])
print(d[0,1,::2])
print(d[0,::-1,-1])
e=d.ravel()
print(e) #展平
print(d.flatten()) #flatten请求内存来保存结果
# d.shape=(6,4)
# print(d)
print(d.transpose(),d.T) #转置

f=arange(9).reshape(3,3)
g=2*f
print(hstack((f,g)))
print(vstack((f,g)))
print(concatenate((f,g),axis=0))
print(dstack((f,g))) #深度组合，沿纵轴
#column_stack,row_stack同hstack、vstack相同

print(hsplit(f,3))
print(vsplit(f,3))
print(split(f,3,axis=0))
print(f.ndim)#维度
print(f.size)
print(f.nbytes,f.size*f.itemsize)

h=array([1.j+1,2.j+3])
print(h.real,h.imag)

#扁平迭代器
i=h.flat
for item in i:
    print(item)
i[[0,1]]=7
i[0]=6
print(h)
print(h.tolist())
h.astype(int)
print(h)
