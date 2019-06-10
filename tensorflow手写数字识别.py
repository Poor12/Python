from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mnist=input_data.read_data_sets("MNIST_data/",one_hot=1)
#我们的图像是28*28大小的图片，展开成1维的结果784维特征
#我们训练数据的特征是55000*784的tensor，第一个维度是图片的编号，第二个维度是图片中像素点的编号，同时训练的数据label是一个55000*10的tensor
#这里对10个种类进行了one-hot编码，label是一个10维向量
#softmax regression
#它将可以判定的特征相加，然后将这些特征转化为判定是这一类的概率，比如对所有所有像素求一个加权值，而权重是模型根据数据自动学习、训练出来，比如某个像素的灰度值大代表很可能是数字n时，这个像素的权重就很大；反之，权重可能是负的
#我们可以将这些特征写成如下公式：i代表第i类，j代表一张图片的第j个像素，bi是bias，顾名思义是数据本身的倾向，比如大部分数字是0，那么0对应的bias就会很大
#featurei=sigma(j)[Wi,j*xj+bi]
#接下来对所有特征求softmax，简单说就是计算一个exp函数，然后标准化，softmax(x)=normalize(exp(x))
#其中判定为第i类的概率：softmax(x)i=exp(xi)/sigma(j)[exp(xj)]

print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784]) #第二个参数代表tensor的shape，也就是数据的尺寸，none表示不限条数的输入

#变量在模型训练迭代中是持久化的，它可以长期存在并且每轮迭代中被更新
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)
#为了训练模型，我们需要定义一个loss function来描述模型对问题的分类精度，loss越小，代表模型的分类结果与真实值的偏差越小，训练的目的是不断对这个loss减小，直到达到一个全局最优或者局部最优
#这里使用cross-entropy
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

#优化算法，随机梯度下降SGD。
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#全局参数初始化器
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})


correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #tf.argmax(y,1)求各个预测的数字中概率最大的一个，tf_argmax(y_,1)则是找样本的真实数字类别
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

