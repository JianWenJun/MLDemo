#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/4/12 下午5:27 
# @Author : ComeOnJian 
# @File : Boosting.py

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np
import time

class ThresholdClass():
    """
    构造一个阈值分类器
    x<v ,val = 1
    x>= -1 ,val = 0
    Adult是二分类问题，当阈值分类器输出val为1表示预测为正例，当阈值分类器输出val为0表示预测为负例
    """
    def __init__(self,train_x,train_y,w):
        """
        初始化阈值分类器
        :param features: 特征集
        :param labels:标签集
        :param w: 样本对应的权重
        """
        self.X = train_x
        self.y = train_y

        self.sample_num = train_x.shape[0]
        #每个样本对应的参数
        self.w = w
        #初始化 feature取值的阈值范围
        self.values = self._get_V_list(self.X)
        self.best_V = -1

    def train(self):
        """
        计算使得分类误差率最小的阈值
        :return: 返回误差率
        """
        best_error_rate = 1.1
        error_rate = 0.0
        for V in self.values:
            #计算每个阈值对应的误差率
            for feature_v_index in range(self.sample_num):
                val = 0
                if self.X[feature_v_index] < V:
                    val = 1
                # error 出现的情况
                if val != self.y[feature_v_index]:
                    error_rate = error_rate + self.w[feature_v_index] * 1

            if error_rate != 0.0 and error_rate < best_error_rate:
                best_error_rate = error_rate
                self.best_V = V

            #没有error的情况
            if best_error_rate == 1.1:
                self.best_V = V
        return best_error_rate

    def predict(self,feature_value):
        """
        :param feature_value:单个维度的值
        :return:预测值
        """
        if feature_value < self.best_V:
            return 1
        else:
            return 0

    def _get_V_list(self,X):
        """
        :param X:特征对应的取值
        :return:该特征对应的阈值可取值列表
        """
        values_set = set(X)
        values = []
        for iter,value in enumerate(values_set):
            if iter==0:
                continue
            else:
                values.append(value-0.5)
        return values

"""
如果特征有n维，我们针对每一维特征求一个分类器，选取这些分类器中分类误差率最低的分类器作为本轮的分类器，将其特征index与分类器一起存入G(x)中。 
"""
class AdaBoostBasic():
    def __init__(self,M = 10):
        # 由M个弱分类器叠加
        self.M = M
        pass
    def _init_parameters_(self,train_x,train_y):
        self.X = train_x
        self.y = train_y
        # 特征数
        self.feature_num = train_x.shape[1]
        self.sample_num = train_x.shape[0]

        # 分类器Gm(x)的系数列表
        self.alpha = []
        # 列表item 为(维度即特征index，维度对应的分类器)
        self.classifier = []

        # 数据集样本的权值分布,类型为列表
        self.w = [1.0/self.sample_num] * self.sample_num

    def train(self,train_x,train_y):
        self._init_parameters_(train_x,train_y)
        #计算M个弱分类器
        for iter in range(self.M):
            print('start %d ThresholdClass ...'%(iter))
            #对于多维度的分类，需要计算特定的维度的最好的分类器即误差率最小
            # 分别对应的误差率，特征index，分类器对象
            best_ThresholdClass = (1,None,None)

            #从多维度的数据中找出维度对应的误差最小的特征及对应的分类器
            for feature_index in range(self.feature_num):
                #取feature对应的列，作为单维特征
                feature_X = self.X[:,feature_index]
                thresholdClass = ThresholdClass(feature_X,self.y,self.w)
                error_rate = thresholdClass.train()

                if error_rate < best_ThresholdClass[0]:
                    best_ThresholdClass = (error_rate,feature_index,thresholdClass)
            error_rate_iter = best_ThresholdClass[0]
            print('No %d ThresholdClass error rate is : %f , feature index is :%d'
                  % (iter,best_ThresholdClass[0],best_ThresholdClass[1]))

            #记录下第iter轮的分类器
            self.classifier.append(best_ThresholdClass)

            # 记录下参数alpha
            alpha_iter = 100
            if error_rate_iter == 1.1:
                #没有分错的情况

                self.alpha.append(alpha_iter)
            else:
                alpha_iter = self._get_alpha(error_rate_iter)
                self.alpha.append(alpha_iter)

            #更新训练集记录的权值分布
            Zm = self._get_Z_m(alpha_iter,best_ThresholdClass[1],best_ThresholdClass[2])
            self._updata_w(alpha_iter,best_ThresholdClass[1],best_ThresholdClass[2],Zm)

    def predict(self,sample):
        predict = 0
        for index in range(self.M):
            alpha_m = self.alpha[index] # 系数
            classfiler_m = self.classifier[index] # 分类器参数
            feature_index_m = classfiler_m[1] #分类器对应的Feature index
            thresholfclass_m = classfiler_m[2]
            feature_value = sample[feature_index_m]
            Gm = thresholfclass_m.predict(feature_value)

            predict = predict + alpha_m * Gm
        predict = self._sigmoid(predict)
        if predict >= 0.5:
            return 1
        else:
            return 0
    def _sigmoid(self,x):
        return 1.0/(1 + math.exp(-x))


    def _get_alpha(self,error_rate_iter):
        alpha = 0.5 * math.log((1-error_rate_iter)/error_rate_iter)
        return alpha
    #规范因子
    def _get_Z_m(self,alpha,feature_index,classifler):
        """
        :param alpha:第m个弱分类前的系数
        :param feature_index 分类的特征的index
        :param classifler:第m个弱分类
        :return:Zm
        """
        Zm = 0.0
        for index in range(self.sample_num):
            temp = - alpha * self.y[index,:][0] * classifler.predict(self.X[index,feature_index])
            Zm = Zm + self.w[index] * math.exp(temp)
        return Zm

    def _updata_w(self,alpha,feature_index,classifler,Zm):
        """更新w值"""
        for index in range(self.sample_num):
            temp = - alpha * self.y[index, :][0] * classifler.predict(self.X[index, feature_index])
            self.w[index] = self.w[index] / Zm * math.exp(temp)
class AdaBoostTree():
    """
    以CART为基类-弱分类器的提升方法
    """
    def __init__(self):
        pass

class AdaBoostGDBT():
    pass

train_file = '../data/adult/adult_deal_value.data'
test_file = '../data/adult/adult_deal_value.test'

if __name__ == '__main__':
    flods = [train_file, test_file]
    print('load data...')
    from ML.DecisionTree import decision_tree as dt
    train_x, train_y, test_x, test_y = dt.load_data(flods)

    print('finish data load...')

    start_time = time.time()
    adboost = AdaBoostBasic(M = 30)
    adboost.train(train_x,train_y)
    end_time = time.time()
    train_time = end_time - start_time
    print('total train time is :%.3f'%train_time)

    pred_y = []
    for sample in test_x:
        pred_yi = adboost.predict(sample)
        pred_y.append(pred_yi)
    # pred_y = [0]* (test_y.shape[0])
    print("accuracy is : ",accuracy_score(y_true=test_y,y_pred=pred_y))