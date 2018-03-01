#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/2/28 下午4:22 
# @Author : ComeOnJian 
# @File : NN_tf.py 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# step 1 NN的参数设置

in_unit = 784
h1_unit = 300

learningrate = 0.05  # 梯度下降法学习率
dropout_keep_prob = 0.75  # dropout时保留神经元的比例，神经网络不为0的参数变为原理的1/dropout_keep_prob倍

batch_size = 100  # 梯度下降法选取的batch的size
max_iter = 3000  # 迭代次数

sava_dir = '../data/'  # 存放数据结果
log_dir = '../log/'  # 日志目录


def variable_summeries(var):
    """
    :param var: Tensor, Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summeries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean) #记录参数的均值

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))

            # 用直方图记录参数的分布
            tf.summary.histogram('histogram',var)

def weight_variable(shape):
    """
    将每一层的神经网络的对应的权重参数w,初始化并封装到function中
    """
    inita_w = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inita_w,dtype=tf.float32)
def bias_variable(shape):
    """
    将每一层的神经网络的对应的偏置项b,初始化并封装到function中
    """
    inita_b = tf.constant(0.1,shape=shape)
    return tf.Variable(inita_b)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
    """
    建立神经网络层（一层）
    :param input_tensor:特征数据
    :param input_dim:输入数据的维度大小
    :param output_dim:该层神经元的个数
    :param layer_name:命名空间
    :param act:神经元对应的激活函数
    """
    #设置命名空间
    with tf.name_scope(layer_name):
        #初始化权重，并记录权重变化
        with tf.name_scope('weights'):
            weight = weight_variable([input_dim,output_dim])
            variable_summeries(weight)# 记录权重变化

        with tf.name_scope('bias'):
            bias = bias_variable([output_dim])
            variable_summeries(bias)

        with tf.name_scope('linear_compute'):
            preact = tf.matmul(input_tensor,weight)+bias
            tf.summary.histogram('linear',preact)

        activeation = act(preact,name = 'activation')
        tf.summary.histogram('activation',activeation)

        return activeation
# def set_computer_Graph():
#
#     """
#     设计tf的计算图，并返回
#     :return:
#     """
#     tf.reset_default_graph()
#     train_graph = tf.Graph()
#
#     with train_graph.as_default():
#
#         # step 3.1 设置算法模型中的输入，使用占位符，占用输入的数据(什么情况下使用占位符，什么情况下设置tf变量)
#
#         train_x = tf.placeholder(dtype=tf.float32,shape=[None,in_unit],name = 'train_x')
#         train_y = tf.placeholder(dtype=tf.float32,shape=[None,10],name = 'train_y')
#
#         # step 3.2构造神经网络
#
#         # 创建第一层隐藏层
#         hidden_layer1 = nn_layer(train_x,input_dim=in_unit,output_dim=h1_unit,layer_name='hider_layer1',act=tf.nn.relu)
#
#         #在第一层隐藏层上创建一层 dropout层 —— 随机关闭一些hidden_layer1的神经元
#         with tf.name_scope('dropout'):
#             dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
#             tf.summary.scalar('dropout_keep_probability',dropout_prob)
#             hidden_layer1_dropout = tf.nn.dropout(hidden_layer1,dropout_prob)
#
#         #创建输出层,包括10个类别，输出层的输入是hidden_layer1_dropout,输出是[1,10]
#         y = nn_layer(hidden_layer1_dropout,h1_unit,10,layer_name='out_layer',act=tf.identity)
#
#         # step 3.3 创建损失函数
#
#         with tf.name_scope('loss'):
#             cross_entropy_diff = tf.nn.softmax_cross_entropy_with_logits(labels=train_y,logits=y)
#
#             with tf.name_scope('total'):
#                 cross_entropy = tf.reduce_mean(cross_entropy_diff)
#         tf.summary.scalar('loss',cross_entropy)
#
#         # step 3.4 选择优化器训练并计算准确率
#         optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
#         train_op = optimizer.minimize(cross_entropy)
#
#         with tf.name_scope('accuracy'):
#             with tf.name_scope('correct_prediction'):
#                 correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(train_y,1))
#             with tf.name_scope('accuracy'):
#                 accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#         tf.summary.scalar('accuracy',accuracy)
#     return train_graph,train_op,accuracy
#
if __name__ == '__main__':
    # step 2 加载数据
    mnist = input_data.read_data_sets('./MNIST_data/',one_hot=True)

    #step 3设置tf 计算图
    tf.reset_default_graph()
    train_graph = tf.Graph()
    with train_graph.as_default():
        # step 3.1 设置算法模型中的输入，使用占位符，占用输入的数据(什么情况下使用占位符，什么情况下设置tf变量)
        train_x = tf.placeholder(dtype=tf.float32,shape=[None,in_unit],name = 'train_x')
        train_y = tf.placeholder(dtype=tf.float32,shape=[None,10],name = 'train_y')

        # step 3.2构造神经网络
        # 创建第一层隐藏层
        hidden_layer1 = nn_layer(train_x,input_dim=in_unit,output_dim=h1_unit,layer_name='hider_layer1',act=tf.nn.relu)

        #在第一层隐藏层上创建一层 dropout层 —— 随机关闭一些hidden_layer1的神经元
        with tf.name_scope('dropout'):
            dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob')
            tf.summary.scalar('dropout_keep_probability',dropout_prob)
            hidden_layer1_dropout = tf.nn.dropout(hidden_layer1,dropout_prob)

        #创建输出层,包括10个类别，输出层的输入是hidden_layer1_dropout,输出是[1,10]
        y = nn_layer(hidden_layer1_dropout,h1_unit,10,layer_name='out_layer',act=tf.identity)

        # step 3.3 创建损失函数
        with tf.name_scope('loss'):
            cross_entropy_diff = tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=y)

            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(cross_entropy_diff)
        tf.summary.scalar('loss', cross_entropy)

        # step 3.4 选择优化器训练并计算准确率
        optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        train_op = optimizer.minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(train_y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)


    session = tf.InteractiveSession(graph=train_graph)

    # step 4 合并summary并初始化所有变量
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir+'/train',graph=train_graph)
    test_writer = tf.summary.FileWriter(log_dir+'/test',graph=train_graph)

    tf.global_variables_initializer().run()

    # Step 5 训练模型并记录到TensorBoard
    for iter in range(max_iter):
        trainx_batch_x,train_batch_y = mnist.train.next_batch(batch_size)
        #迭代10次记录一下accuracy
        if iter % 10 == 0:
            summmary,acc,loss = session.run([merged,accuracy,cross_entropy],feed_dict={train_x:trainx_batch_x,train_y:train_batch_y,dropout_prob:1.0})
            test_writer.add_summary(summmary,iter)#写入日志
            print('loss at step %s: %s'%(iter,loss))
            print('Accuracy at step %s: %s'%(iter,acc))
        else:
            if iter % 100 == 0:
                #记录tensor运行节点的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                #将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
                summmary,_ = session.run([merged,train_op],
                                         feed_dict={train_x:trainx_batch_x,train_y:train_batch_y,dropout_prob:dropout_keep_prob},
                                         options=run_options,
                                         run_metadata=run_metadata)
                #将节点运行时的信息写入日志文件
                train_writer.add_run_metadata(run_metadata,'step %d' % iter)
                train_writer.add_summary(summmary,iter)
                pass
            else:
                summmary,_ = session.run([merged,train_op],feed_dict={train_x:trainx_batch_x,train_y:train_batch_y,dropout_prob:dropout_keep_prob})
                train_writer.add_summary(summmary,iter)
    train_writer.close()
    test_writer.close()
    session.close()

