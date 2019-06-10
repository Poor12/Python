from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#自编码器，可以使用自身的高阶特征编码自己
#自编码器借助稀疏编码的思想，目标是使用稀疏的一些高阶特征重新组合来重构自己
#特点有二：1.期望输入\输出一致；2.希望使用高阶特征来重构自己
#如果限制中间隐含层的数量，就相当于一个降维的过程。
#如果给数据添加噪声，那么就是去噪自编码器，我们将会从噪声中学习出数据的特征
#去噪自编码器中最常使用的噪声是加性高斯噪声，AGN，也可以使用随机遮挡的噪声

import numpy as np
import sklearn.preprocessing as prep
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#xaiver初始化器,就是让权重满足0均值，同时方差为2/(nin+nout)
def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input #输入层
        self.n_hidden=n_hidden #隐含层
        self.transfer=transfer_function #隐含层激活函数，默认为softplus
        self.scale=tf.placeholder(tf.float32) #高斯噪音
        self.training_scale=scale #高斯噪音系数
        network_weights=self._initialize_weights()
        self.weights=network_weights

        #定义网络结构
        #为输入x创建一个维度为n_input的placeholder，然后建立一个能提取特征的隐含层
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #定义自编码器的损失函数
        #平方误差
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        #先创建一个名为all_weights的字典dict，然后将w1、b1、w2、b2全部存入其中，然后返回all_weights
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    #定义计算损失cost及执行一步训练的函数partial_fit
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X})
        return cost

    #只求损失
    def cal_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X})

    #返回自编码器隐含层的输出结果，它的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能是学习出数据中的高阶特征
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X})

    #将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据，这个接口和前面的transfrom正好将整个自编码器拆分为两部分
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X})

    #获取隐含层权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #获取隐含层的偏置系数b1
    def getBias(self):
        return self.sess.run(self.weights['b1'])


mnist=input_data.read_data_sets('MNIST_data',one_hot=1)

#定义一个对训练、测试数据进行标准化处理的函数
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

#定义一个获取随机block数据的函数：取一个从0到len(data)-batch_size之间的随机整数，再以这个随机数作为block的起始位置，然后顺序取到一个batch_size的数据
def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)

#定义常用参数，总训练样本数，最大训练的轮数设为20，batch_size设为128，并设置每隔一轮就显示一次损失
n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

#创建一个AGN自编码器的实例，定义模型输入节点数n_input为784，自编码器的隐含层节点数n_hidden为200，隐含层的激活函数transfer_function为softplus，优化器op为Adam且学习速率为0.001，同时噪声的系数scale设为0.01
autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)


#在每一轮循环开始时，我们将平均avg_cost设为0，并计算总共需要的batch数，然后在每一个batch的循环中，先随机抽取一个block的数据，然后使用成员函数训练这个batch的数据并计算当前的cost，最后将当前的cost整合到avg_cost中
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=autoencoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size

    if epoch % display_step ==0:
        print("Epoch: ",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))

print("Total cost: "+str(autoencoder.cal_total_cost(X_test)))