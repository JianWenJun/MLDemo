#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/23 下午8:47 
# @Author : ComeOnJian 
# @File : tree_main.py

from ML.DecisionTree import  decision_tree as dt
import time
import numpy as np
from sklearn.metrics import accuracy_score

train_file = '../data/adult/adult.data'
test_file = '../data/adult/adult.test'
def test():
    node = dt.TreeNode()
    node.add_attr('name','jianwenjun')
    # print (node.name)
    a = []
    a.append(node)
    node1 =  dt.TreeNode()
    node1.value =12
    a.append(node1)
    print(a[0].name)
    print(a[1].value)


if __name__ == '__main__':
    flods = [train_file,test_file]
    print('load data...')
    train_x, train_y, test_x, test_y = dt.load_data(flods)
    print('finish data load...')
    start_time = time.time()
    # print (dt.calc_ent(train_y))
    # print (dt.calc_condition_ent(train_x[:,3],train_y))
    # print (test_y[429,0])
    # print (type(train_y[0][0]))
    # train
    # print(type(test_y[:,0]),test_y[:,0].shape)




    # decision_tree = dt.DecisionTree(mode='ID3')
    # print('train start ------')
    # decision_tree.train(train_x,train_y,0.1)
    # print('train finish ------')
    # print('predict start ------')
    #
    # predict_y = decision_tree.predict(test_x)
    #
    #
    # print('accuracy: %f' %(accuracy_score(y_true=test_y,y_pred=predict_y)))

