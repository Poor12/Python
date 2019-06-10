import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "Model/model.ckpt")

# #直接加载已持久化的图
# saver=tf.train.import_meta_graph("Model/model.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess,"./Model/model.ckpt")
#     print(sess.run(result))
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

u1=tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
u2=tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")
result=u1+u2

saver=tf.train.Saver({"v1":u1,"v2":u2})
with tf.Session() as sess:
    saver.restore(sess,"./Model/model.ckpt")
    print(sess.run(result))

#保存滑动平均模型
v=tf.Variable(0,dtype=tf.float32,name="v")
for variables in tf.global_variables():
    print(variables.name)
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op=ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name)

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    saver.save(sess,"Model/model_ema.ckpt")
    print(sess.run([v,ema.average(v)]))

#通过变量重命名直接读取变量的滑动平均值
v=tf.Variable(0,dtype=tf.float32,name="v")
saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"./Model/model_ema.ckpt")
    print(sess.run(v))

#通过将计算图中的变量及其取值通过常量方式保存起来
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #导出当前计算图的GraphDef部分，即从输入层到输出层的计算过程部分
    graph_def=tf.get_default_graph().as_graph_def()
    output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['add'])
    with tf.gfile.GFile("Model/combined_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())

with tf.Session() as sess:
    model_filename="Model/combined_model.pb"
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result=tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))