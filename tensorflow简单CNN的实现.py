from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#一般卷积神经网络由多个卷积层构成，每个卷积层通常会进行如下几个操作
#1.图像通过多个不同的卷积核的滤波，并加偏置，提取出局部特征，每一个卷积核会映射出一个新的2d图像
#2.将前面卷积核的滤波输出结果，进行非线性的激活函数处理，目前最常见的是使用ReLU函数，而以前sigmod使用较多
#3.对激活函数的结果进行池化操作，目前一般使用max pooling，保留最显著的特征，并提交模型的畸变容忍能力

#一个卷积层中可以有多个不同的卷积核，而每个卷积核对应一个滤波后映射出的新图像，同一个新图像中每个像素都来自完全相同的卷积核，这就是卷积核的权值共享
#将全连接变为局部连接可进一步降低参数量
#每一个卷积核滤波得到的图像是一类特征的映射，即一个feature map
#卷积的好处是不管图片尺寸如何，我们需要训练的权值数量只和卷积核大小、卷积核数量相关，我们可以使用非常少的参数量处理任意大小的图片
#每一个卷积层提取的特征，在后面的层中都会抽象组合成更高阶的特征，而且多层抽象的卷积网络表达能力更强，效率更高
#隐含节点的数量只跟卷积的步长有关，如果步长为1，核的大小为1，那么隐含节点的数量与输入的图像像素数量一致
#CNN的要点：1.局部连接 2.权值共享 3.池化的降采样

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

mnist=input_data.read_data_sets("MNIST_data/",one_hot=1)
sess=tf.InteractiveSession()

#给权重增加一些随机噪声来打破完全对称，比如截断的正态分布噪音
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#因为使用ReLU，也给偏置增加一些正值用来避免死亡节点
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#2维卷积函数，x是输入，W是卷积的参数，比如[5,5,1,32]，5*5表示卷积核的大小，1表示信道数，因为图片用灰度表示所以只有1，最后一个表示卷积核的数量，padding表示边界的处理方法
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#最大池化函数，2*2的像素降为1*1，最大池化会保留原始像素块中灰度值最高的那一个像素
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义输入的placeholder，x是特征，y_是真实的label
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

#-1代表样本数量不固定，1代表颜色通道数量
x_image=tf.reshape(x,[-1,28,28,1])

#定义第一个卷积层
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool(h_conv1)

#定义第二个卷积层
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool(h_conv2)

#7*7*64
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#为了减轻过拟合，下面使用一个dropout层，训练时随机丢弃一部分数据来减轻过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#最后将dropout层的输出连接一个softmax层，得到最后的概率输出
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#定义损失函数，但是优化器是Adam，给予1e-4
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义评测准确率的操作
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化所有参数，设置训练时dropout的keep_prob比率为0.5，然后使用大小为50的mini-batch，共进行20000次训练迭代，参与训练的样本数量总共为100w，其中每100次训练进行评测
tf.global_variables_initializer().run()
for i in range(1000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
