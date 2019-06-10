import tensorflow as tf
#tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)
#第一个参数指需要做卷积的输入图像，它要求是一个Tensor，具有[训练一个batch的图片数量，图片高度，图片宽度，图像通道]
#第二个参数指CNN中的卷积核,[卷积核高度，卷积核的宽度，图像通道数，卷积核个数]
#第三个参数指卷积是在每一维的步长，是一个一维的向量，长度为4
#第四个参数指string类型，只能是"SAME"和"VALID"
oplist=[]
input_arg=tf.Variable(tf.ones([1,3,3,5]))
filter_arg=tf.Variable(tf.ones([1,1,5,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 2"])

input_arg=tf.Variable(tf.ones([1,3,3,5]))
filter_arg=tf.Variable(tf.ones([3,3,5,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 4"])

input_arg=tf.Variable(tf.ones([1,5,5,5]))
filter_arg=tf.Variable(tf.ones([3,3,5,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 5"])

#1*3*3*7
input_arg=tf.Variable(tf.ones([1,5,5,5]))
filter_arg=tf.Variable(tf.ones([3,3,5,7]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 6"])

#SAME可停留在边缘，1*5*5*7
input_arg=tf.Variable(tf.ones([1,5,5,5]))
filter_arg=tf.Variable(tf.ones([3,3,5,7]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 7"])

#输入4张图，4*5*5*7
input_arg=tf.Variable(tf.ones([4,5,5,5]))
filter_arg=tf.Variable(tf.ones([3,3,5,7]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 8"])

#1*1*100*60
input_arg=tf.Variable(tf.ones([1,1,100,3]))
filter_arg=tf.Variable(tf.ones([1,10,3,60]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 9"])

#bias对每个值就进行计算
input_arg=tf.Variable(tf.ones([1,1,100,3]))
filter_arg=tf.Variable(tf.ones([1,10,3,60]))
biases = tf.get_variable("bias",[60],initializer=tf.constant_initializer(0.1))
op2 = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID'),biases)),ksize=[1,1,20,1],strides=[1,1,1,1],padding='VALID')
oplist.append([op2, "case 10"])

#1*2*3*1
input_arg=tf.Variable(tf.ones([1,3,3,5]))
filter_arg=tf.Variable(tf.ones([1,1,5,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 11"])

#
input_arg=tf.Variable(tf.ones([600,1,50,60]))
filter_arg=tf.Variable(tf.ones([1,6,60,10]))

op2 = tf.nn.relu(tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID'))
oplist.append([op2, "case 12"])
print(op2.get_shape().as_list())

#maxpool
#tf.nn.max_pool(value,ksize,strides,padding,name=None)
#第一个参数：需要池化的输入，[batch,height,width,channels]
#第二个参数，池化窗口的大小，同上
#第三个参数：窗口在每个维度上滑动的的步长
#第四个参数：可以去valid或same

a=tf.constant([
    [[1.0,2.0,3.0,4.0],
     [5.0,6.0,7.0,8.0],
     [8.0,7.0,6.0,5.0],
     [4.0,3.0,2.0,1.0]],
    [[4.0,3.0,2.0,1.0],
     [8.0,7.0,6.0,5.0],
     [1.0,2.0,3.0,4.0],
     [5.0,6.0,7.0,8.0]]
])
a=tf.reshape(a,[1,4,4,2])
pooling=tf.nn.max_pool(a,[1,2,2,1],[1,2,1,1],padding='VALID')
oplist.append([pooling,"case 13"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer())
    for aop in oplist:
        print("------------{}-------------".format(aop[1]))
        result=a_sess.run(aop[0])
        print(result)
        print(result.shape)
        print("------------------------\n\n")
