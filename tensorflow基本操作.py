#tensorflow基本使用
#使用图来表示计算任务
#被称为会话的上下文中执行图
#使用tensor表示数据
#使用变量维护状态
#使用feed和fetch可以为任意的操作赋值或从其中获取数据

#tensorflow是一个编程系统，使用图来表示计算任务，图中的节点被称为op，一个op获取0或多个tensor，产生0或多个tensor，每个tensor是一个多类型的多维数据
#一个tensorflow图描述了计算的过程，为了进行计算，图必须在会话中启动，会话把资源分配给诸如CPU或者GPU的设备并执行op的方法，这些方法执行后返回numpy ndarray数组

#tensorflow程序通常被组织成一个构建阶段和执行阶段，构建阶段，op的执行步骤被描述成一个图；执行阶段，使用会话执行执行图里的op
#构建图的第一步，是创建源op，源op不需任何输入，例如常量，源op的输出被传递给其他op做运算
#op构造器的返回值代表被构造出的op的输出，这些返回值可以传递给其他op构造器作为输入
#python库有一个默认图，op构造器可以为它增加节点

#tensorflow内建的运算操作
#1.标量运算
#2.向量运算
#3.矩阵运算
#4.带状态的运算
#5.神经网络组件
#6.储存和恢复
#7.队列及同步运算
#8.控制流
import tensorflow as tf

#构建阶段
matrix1=tf.constant([[3.,3.]]) #1*2
matrix2=tf.constant([[2.],[2.]]) #2*1

product=tf.matmul(matrix1,matrix2)

#执行阶段
#sess=tf.Session()
#result=sess.run(product)
#print(result)

#显示关闭
#sess.close()

#隐式关闭
#with tf.Session() as sess:
#    result=sess.run([product])
#    print(result)

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        result=sess.run([product])
        print(result)

#交互式使用，避免使用一个变量来持有会话
#使用tensor.eval()和operation.run()来代替session.run()
sess=tf.InteractiveSession()
x=tf.Variable([1.0,2.0])
a=tf.constant([3.0,3.0])
print(a.shape)#维度
print(a._rank())
print(a.dtype)#数据类型
x.initializer.run()
#增加一个减法sub op，从x减去a，运行减法op，输出结果
sub=tf.subtract(x,a)
print(sub.eval())


#变量维护图执行过程中的状态信息
#创建一个变量，初始化为标量0
state=tf.Variable(0,name="counter")
#创建一个op，其作用是使state增加1
one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value) #赋值操作

#启动图后变量必须先进行初始化
#首先必须增加一个初始化op到图中
init_op=tf.initialize_all_variables()
#启动图，运行op
with tf.Session() as sess:
    #运行init op
    sess.run(init_op)
    #打印state的初值
    print(sess.run(state))
    #运行op，更新state，打印'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
intermed=tf.add(input2,input3)
mul=tf.multiply(input1,intermed)

with tf.Session() as sess:
    result=sess.run([mul,intermed]) #在op的一次运行中获得
    print(result)

#feed使用一个tensor值临时替换一个操作的输出结果，你可以提供feed数据作为run()调用的参数
#feed只在调用它的方法内有效，方法结束，feed就会消失
input1=tf.placeholder(tf.dtypes.float32)
input2=tf.placeholder(tf.dtypes.float32)
output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))

