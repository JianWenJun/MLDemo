#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/23 下午8:47 
# @Author : ComeOnJian 
# @File : tree_main.py

from ML.DecisionTree import  decision_tree as dt
import time
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

train_file = '../data/adult/adult_deal_value.data'
test_file = '../data/adult/adult_deal_value.test'

if __name__ == '__main__':
    flods = [train_file,test_file]
    print('load data...')
    train_x, train_y, test_x, test_y = dt.load_data(flods)

    # print(type(train_x[:,0][0]))
    print('finish data load...')

    my_decision_tree = dt.DecisionTree(mode='ID3')
    my_decision_tree.train(train_x,train_y,0.01)
    predict_y = my_decision_tree.predict(test_x)
    print('my tree accuracy: %f' % (accuracy_score(y_true=test_y, y_pred=predict_y)))


    decisiont_tr = DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=9)
    decisiont_tr.fit(train_x,train_y)
    p_y =  decisiont_tr.predict(test_x)
    print('sklearn tree accuracy: %f' % (accuracy_score(y_true=test_y, y_pred=p_y)))
    print(decisiont_tr.feature_importances_)



