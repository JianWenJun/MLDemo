#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/23 下午8:47 
# @Author : ComeOnJian 
# @File : tree_main.py

from ML.DecisionTree import  decision_tree as dt
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




# np.array()

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


    decision_tree = dt.DecisionTree(mode='ID3')
    print('train start ------')
    decision_tree.train(train_x,train_y,0.1)
    print('train finish ------')
    print('predict start ------')

    predict_y = decision_tree.predict(test_x)
    # print('predict_y: ',len(predict_y))
    # print('test_y: ',len(test_y))

    print('accuracy: %f' %(accuracy_score(y_true=test_y,y_pred=predict_y)))
    print('average_precision_score: %f' %(average_precision_score(y_true=test_y,y_score=predict_y)))

    print('MMC: %f'%(matthews_corrcoef(y_true=test_y,y_pred=predict_y)))

    print(classification_report(y_true=test_y, y_pred=predict_y))
    from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score

    # Calculate precision score
    print (precision_score(test_y, predict_y, average='macro'))
    print (precision_score(test_y, predict_y, average='micro'))
    print (precision_score(test_y, predict_y, average=None))

    # Calculate recall score
    print(recall_score(test_y, predict_y, average='macro'))
    print(recall_score(test_y, predict_y, average='micro'))
    print(recall_score(test_y, predict_y, average=None))

    # Calculate f1 score
    print(f1_score(test_y, predict_y, average='macro'))
    print(f1_score(test_y, predict_y, average='micro'))
    print(f1_score(test_y, predict_y, average=None))

    fpr, tpr, thresholds = roc_curve(test_y, predict_y)
    roc_auc = auc(fpr, tpr)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc))
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("Receiver Operating Characteristic, ROC(AUC = %0.2f)" % (roc_auc))
    plt.show()

    # y = np.asarray(test_y)
    # print(y.dtype)
    # print(isinstance(y.flat[0], str))
    # if y.ndim > 2 or (y.dtype == object and len(y) and
    #                   not isinstance(y.flat[0], str)):
    #     print("unknow")

    # for sample in test_x:
    #     print(sample)
    #     sample_var = sample.reshape(1,14)[0,:]
    #     print(sample_var)





    # decision_tree.saveModel('test.dt')
    #
    # predict
    # test_pred = decision_tree.predict(test_x)
    #
    # computer accuracy
    # score = accuracy_score(y_true=test_y,y_pred=test_pred)

    # print (dt.calc_condition_gini(train_x[:,0],train_y,39))
    #calcul
    # count = np.bincount(train_y[:,0])
    # max_value = np.argmax(count)
    # print(max_value)

    # test1()
    # D = train_y[]


