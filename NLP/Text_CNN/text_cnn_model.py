#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/9 下午4:35 
# @Author : ComeOnJian 
# @File : text_cnn_model.py 

#  structure TextCNN
#  1.input-embedding layer(max_l * 300) 2.cov layer 3.max pool layer 4.full connect droput + softmax + l2

import tensorflow as tf
import numpy as np
import pdb

class TextCNN():

    __shuffer_falg = False
    __static_falg = True

    def __init__(self,shuffer_falg, static_falg, filter_numbers, filter_sizes, sentence_length,embedding_size,learnrate, epochs, batch_size, dropout_pro):

        self.__shuffer_falg = shuffer_falg
        self.__static_falg = static_falg
        self.learning_rate_item = learnrate
        self.epochs = epochs
        self.sentence_length = sentence_length
        self.filter_numbers = filter_numbers
        self.batch_size = batch_size
        self.dropout_pro_item = dropout_pro
        self.embedding_size = embedding_size
        # 1. setting graph
        tf.reset_default_graph()
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # self.input_x = tf.placeholder(dtype=tf.float32,shape=[None,sentence_length],name='input_x')
            # 1 input layer
            self.embedding_layer = tf.placeholder(dtype=tf.float32, shape=[None, sentence_length, embedding_size],
                                                  name='embeding_layer')
            if self.__static_falg:
                # the word vectors are not allowed to change
                self.embedding_layer_expand = tf.expand_dims(self.embedding_layer,-1)#[None,sentence_length,embedding_size,1]
            else:
                self.unstatic_embedding_layer = tf.Variable(self.embedding_layer,name='embedding_layer_unstatic')
                self.embedding_layer_expand = tf.expand_dims(self.unstatic_embedding_layer,-1)
            self.input_y = tf.placeholder(dtype=tf.int32,shape=[None,2],name='input_y')
            self.dropout_pro = tf.placeholder(dtype=tf.float32,name='dropout_pro')
            self.learning_rate = tf.placeholder(dtype=tf.float32,name='learning_rate')
            self.l2_loss = tf.constant(0.0)

            #2 embedding layer

            # with tf.name_scope('word_embedding_layer'):
            #
            #     embedding_matrix = tf.Variable()
            #     tf.expand_dims()

            #3 conv layer + maxpool layer for each filer size
            pool_layer_lst = []
            for filter_size in filter_sizes:
                max_pool_layer = self.__add_conv_layer(filter_size,filter_numbers)
                pool_layer_lst.append(max_pool_layer)

            # 4.full connect droput + softmax + l2
            # combine all the max pool —— feature

            with tf.name_scope('dropout_layer'):
                # pdb.set_trace()

                max_num = len(filter_sizes) * self.filter_numbers
                h_pool = tf.concat(pool_layer_lst,name='last_pool_layer',axis=3)
                pool_layer_flat = tf.reshape(h_pool,[-1,max_num],name='pool_layer_flat')

                dropout_pro_layer = tf.nn.dropout(pool_layer_flat,self.dropout_pro,name='dropout')

            with tf.name_scope('soft_max_layer'):
                SoftMax_W = tf.Variable(tf.truncated_normal([max_num,2],stddev=0.01),name='softmax_linear_weight')
                self.__variable_summeries(SoftMax_W)
                # print('test1------------')
                SoftMax_b = tf.Variable(tf.constant(0.1,shape=[2]),name='softmax_linear_bias')
                self.__variable_summeries(SoftMax_b)
                # print('test2------------')
                self.l2_loss += tf.nn.l2_loss(SoftMax_W)
                self.l2_loss += tf.nn.l2_loss(SoftMax_b)
                # dropout_pro_layer_reshape = tf.reshape(dropout_pro_layer,[batch_size,-1])
                self.softmax_values = tf.nn.xw_plus_b(dropout_pro_layer,SoftMax_W,SoftMax_b,name='soft_values')
                # print ('++++++',self.softmax_values.shape)
                self.predictions = tf.argmax(self.softmax_values,axis=1,name='predictions',output_type=tf.int32)

            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_values,labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + 0.001 * self.l2_loss #lambda = 0.001

                # print ('---------1',self.loss)
                tf.summary.scalar('last_loss',self.loss)

            with tf.name_scope('accuracy'):
                correct_acc = tf.equal(self.predictions,tf.argmax(self.input_y,axis=1,output_type=tf.int32))

                self.accuracy = tf.reduce_mean(tf.cast(correct_acc,'float'),name='accuracy')
                tf.summary.scalar('accuracy',self.accuracy)

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # print('test1------------')
                # pdb打个断点
                # pdb.set_trace()
                self.train_op = optimizer.minimize(self.loss)
                # print('test2------------')

            self.session = tf.InteractiveSession(graph=self.train_graph)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./NLP/log/text_cnn', graph=self.train_graph)
    def train(self,train_x,train_y):
        self.session.run(tf.global_variables_initializer())

        #迭代训练
        for epoch in range(self.epochs):
            # pdb.set_trace()
            train_batch = self.__get_batchs(train_x, train_y, self.batch_size)
            train_loss, train_acc, count = 0.0, 0.0, 0
            for batch_i in range(len(train_x)//self.batch_size):
                x,y = next(train_batch)
                # print('--------',np.array(x).shape)
                feed = {
                    self.embedding_layer:x,
                    self.input_y:y,
                    self.dropout_pro:self.dropout_pro_item,
                    self.learning_rate:self.learning_rate_item
                }

                _,summarys,loss,accuracy = self.session.run([self.train_op,self.merged,self.loss,self.accuracy],feed_dict=feed)
                train_loss, train_acc, count = train_loss + loss, train_acc + accuracy, count + 1
                self.train_writer.add_summary(summarys,epoch)
                # each 5 batch print log
                if (batch_i+1) % 15 == 0:
                    print('Epoch {:>3} Batch {:>4}/{} train_loss = {:.3f} accuracy = {:.3f}'.
                          format(epoch,batch_i,(len(train_x)//self.batch_size),train_loss/float(count),train_acc/float(count)))

    def validataion(self,test_x, test_y):
        test_batch = self.__get_batchs(test_x,test_y,self.batch_size)
        eval_loss, eval_acc ,count= 0.0, 0.0 ,0
        for batch_i in range(len(test_x) // self.batch_size):
            x,y = next(test_batch)
            feed = {
                self.embedding_layer: x,
                self.input_y: y,
                self.dropout_pro: self.dropout_pro_item,
                self.learning_rate: 1.0
            }
            loss ,accuracy = self.session.run([self.loss,self.accuracy],feed_dict=feed)
            eval_loss ,eval_acc ,count  = eval_loss+loss ,eval_acc+accuracy ,count+1
            # print('validataion_{}_accuracy is {:.3f}'.format(index,accuracy))
        return eval_acc/float(count),eval_loss/float(count)
    def close(self):
        self.session.close()
        self.train_writer.close()
    #generate batch data
    def __get_batchs(self,Xs,Ys,batch_size):
        for start in range(0,len(Xs),batch_size):
            end = min(start+batch_size,len(Xs))
            yield Xs[start:end],Ys[start:end]
        pass
    def __add_conv_layer(self,filter_size,filter_num):
        with tf.name_scope('conv-maxpool-size%d'%(filter_size)):
            #convolutio layer
            filter_shape =[filter_size,self.embedding_size,1,filter_num]
            W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='filter_weight')
            self.__variable_summeries(W)
            b = tf.Variable(tf.constant(0.1,shape=[filter_num]),name='filter_bias')
            self.__variable_summeries(b)
            #参数说明
            #第一个参数input：指需要做卷积的输入图像 [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
            #第二个参数filter：相当于CNN中的卷积核 [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
            #第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4,
            #第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
            #第五个参数：use_cudnn_on_gpu: bool类型，是否使用cudnn加速，默认为true
            conv_layer = tf.nn.conv2d(self.embedding_layer_expand,W,strides=[1,1,1,1],padding='VALID',name='conv_layer')
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,b),name='relu_layer')

            max_pool_layer = tf.nn.max_pool(relu_layer,ksize=[1,self.sentence_length - filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='maxpool')
            return max_pool_layer

    def __variable_summeries(self,var):
        """
        :param var: Tensor, Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summeries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)  # 记录参数的均值

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

                # 用直方图记录参数的分布
                tf.summary.histogram('histogram', var)
