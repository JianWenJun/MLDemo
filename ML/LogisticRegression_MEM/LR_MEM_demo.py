#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/5/16 上午11:44 
# @Author : ComeOnJian 
# @File : LR_MEM_demo.py 

import numpy as np
import pandas as pd
import random
import re
import copy
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

train_file = '../data/Titanic/train.csv'
test_file = '../data/Titanic/test.csv'
test_result_file = '../data/Titanic/gender_submission.csv'

def data_feature_engineering(full_data,age_default_avg=True,one_hot=True):
    """
    :param full_data:全部数据集包括train,test
    :param age_default_avg:age默认填充方式，是否使用平均值进行填充
    :param one_hot: Embarked字符处理是否是one_hot编码还是映射处理
    :return: 处理好的数据集
    """
    for dataset in full_data:
        # Pclass、Parch、SibSp不需要处理

        # sex 0,1
        dataset['Sex'] = dataset['Sex'].map(Passenger_sex).astype(int)

        # FamilySize
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        # IsAlone
        dataset['IsAlone'] = 0
        isAlone_mask = dataset['FamilySize'] == 1
        dataset.loc[isAlone_mask, 'IsAlone'] = 1

        # Fare 离散化处理，6个阶段
        fare_median = dataset['Fare'].median()
        dataset['CategoricalFare'] = dataset['Fare'].fillna(fare_median)
        dataset['CategoricalFare'] = pd.qcut(dataset['CategoricalFare'],6,labels=[0,1,2,3,4,5])

        # Embarked映射处理，one-hot编码,极少部分缺失值处理
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Embarked'] = dataset['Embarked'].astype(str)
        if one_hot:
            # 因为OneHotEncoder只能编码数值型，所以此处使用LabelBinarizer进行独热编码
            Embarked_arr = LabelBinarizer().fit_transform(dataset['Embarked'])
            dataset['Embarked_0'] = Embarked_arr[:, 0]
            dataset['Embarked_1'] = Embarked_arr[:, 1]
            dataset['Embarked_2'] = Embarked_arr[:, 2]
            dataset.drop('Embarked',axis=1,inplace=True)
        else:
            # 字符串映射处理
            dataset['Embarked'] = dataset['Embarked'].map(Passenger_Embarked).astype(int)

        # Name选取称呼Title_name
        dataset['TitleName'] = dataset['Name'].apply(get_title_name)
        dataset['TitleName'] = dataset['TitleName'].replace('Mme', 'Mrs')
        dataset['TitleName'] = dataset['TitleName'].replace('Mlle', 'Miss')
        dataset['TitleName'] = dataset['TitleName'].replace('Ms', 'Miss')
        dataset['TitleName'] = dataset['TitleName'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                            'Other')
        dataset['TitleName'] = dataset['TitleName'].map(Passenger_TitleName).astype(int)

        # age —— 缺失值，分段处理
        if age_default_avg:
            # 缺失值使用avg处理
            age_avg = dataset['Age'].mean()
            age_std = dataset['Age'].std()
            age_null_count = dataset['Age'].isnull().sum()
            age_default_list = np.random.randint(low=age_avg - age_std, high=age_avg + age_std, size=age_null_count)

            dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_default_list
            dataset['Age'] = dataset['Age'].astype(int)
        else:
            # 将age作为label，预测缺失的age
            # 特征为 TitleName,Sex,pclass,SibSP,Parch,IsAlone,CategoricalFare,FamileSize,Embarked
            feature_list = ['TitleName', 'Sex', 'Pclass', 'SibSp', 'Parch', 'IsAlone','CategoricalFare',
                            'FamilySize', 'Embarked','Age']
            if one_hot:
                feature_list.append('Embarked_0')
                feature_list.append('Embarked_1')
                feature_list.append('Embarked_2')
                feature_list.remove('Embarked')
            Age_data = dataset.loc[:,feature_list]

            un_Age_mask = np.isnan(Age_data['Age'])
            Age_train = Age_data[~un_Age_mask] #要训练的Age

            # print(Age_train.shape)
            feature_list.remove('Age')
            rf0 = RandomForestRegressor(n_estimators=60,oob_score=True,min_samples_split=10,min_samples_leaf=2,
                                                                   max_depth=7,random_state=10)

            rf0.fit(Age_train[feature_list],Age_train['Age'])

            def set_default_age(age):
                if np.isnan(age['Age']):
                    # print(age['PassengerId'])
                    # print age.loc[feature_list]
                    data_x = np.array(age.loc[feature_list]).reshape(1,-1)
                    # print data_x
                    age_v = round(rf0.predict(data_x))
                    # print('pred:',age_v)
                    # age['Age'] = age_v
                    return age_v
                    # print age
                return age['Age']

            dataset['Age'] = dataset.apply(set_default_age, axis=1)
            # print(dataset.tail())
            #
            # data_age_no_full = dataset[dataset['Age'].]

        # pd.cut与pd.qcut的区别，前者是根据取值范围来均匀划分，
        # 后者是根据取值范围的各个取值的频率来换分，划分后的某个区间的频率数相同
        # print(dataset.tail())
        dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5,labels=[0,1,2,3,4])
    return full_data
def data_feature_select(full_data):
    """
    :param full_data:全部数据集
    :return:
    """
    for data_set in full_data:
        drop_list = ['PassengerId','Name','Age','Fare','Ticket','Cabin']
        data_set.drop(drop_list,axis=1,inplace=True)
    train_y = np.array(full_data[0]['Survived'])
    train = full_data[0].drop('Survived',axis=1,inplace=False)
    # print(train.head())
    train_X = np.array(train)
    test_X = np.array(full_data[1])
    return train_X,train_y,test_X
def Passenger_sex(x):
    sex = {'female': 0, 'male': 1}
    return sex[x]
def Passenger_Embarked(x):
    Embarked = {'S': 0, 'C': 1 , 'Q': 2}
    return Embarked[x]
def Passenger_TitleName(x):
    TitleName = {'Mr': 0, 'Miss': 1, 'Mrs': 2,'Master': 3, 'Other': 4}
    return TitleName[x]
def get_title_name(name):
    title_s = re.search(' ([A-Za-z]+)\.', name)
    if title_s:
        return title_s.group(1)
    return ""

class LR:
    def __init__(self,iterNum = 2000,learn_late = 0.005):
        self.maxIter = iterNum
        self.learn_late = learn_late

    def train(self,train_X,train_y):
        feature_size = train_X.shape[1]
        sample_size = train_X.shape[0]
        # 将w,b 融合了
        self.w = np.zeros(feature_size + 1)



        correct_num = 0
        #梯度下降算法
        for iter in range(self.maxIter):
            # 随机选取一个样本
            sample_index = random.randint(0,sample_size-1)
            sample_select = train_X[sample_index].tolist()
            sample_select.append(1.0)
            sample_y = train_y[sample_index]

            if sample_y == self.predict(sample_select):
                # 连续预测对一定的数量
                correct_num = correct_num + 1
                if correct_num > self.maxIter:
                    break
                continue
            correct_num = 0
            temp = np.exp(sum(self.w * sample_select))

            for index in range(feature_size):
                self.w[index] = self.w[index] - self.learn_late * \
                                (- sample_y * sample_select[index] + float(temp * sample_select[index])/float(1 + temp))

    def predict(self,sample):
        # 《统计学习方法》公式6.3、6.4
        tmp = sum(self.w * sample)
        y_0 = 1 / float(1+np.exp(tmp))
        y_1 = np.exp(tmp) / float(1+np.exp(tmp))
        if y_0 > y_1:
            return 0
        else:
            return 1

class MEM:
    # 算法模型为：《统计学习方法》公式6.28&6.29

    def __init__(self,iterNum = 2000,epsion = 0.01):
        self.epsion = epsion # 精度阈值
        self.maxIter = iterNum

    def train(self,train_X,train_y):
        # 使用《统计学习方法》P92算法6.2——BFGS，求解参数
        self.feature_size = train_X.shape[1]
        self.sample_num = train_X.shape[0]

        self.samples = train_X
        self.labels = train_y

        # 统计数据集中的特征函数个数
        self._cal_feature_func()
        self._f2id()
        self.n = len(self.P_x_y) # n为特征函数的个数
        # 计算每个特征函数关于经验分布p(x,y)的期望，并保持于EPxy字典中
        self._cal_EPxy()

        self.w = np.zeros(self.n) #wi为拉格函数中的乘子
        self.g = np.zeros(self.n) #对应g(w),《统计学习方法》P92,最上面g(w)的公式

        self.B = np.eye(self.n) #正定对称矩阵

        for iter in range(self.maxIter):

            # 算法6.2——(2）
            self._cal_Gw()
            if self._cal_g_l2() < self.epsion:
                break
            # 算法6.2——(3）
            p_k = - (self.B ** -1) * np.reshape(self.g,(self.n,1))

            # np.linalg.solve()
            # 算法6.2——(4）
            r_k = self._cal_fw()

            # 算法6.2——(5）
            old_g = copy.deepcopy(self.g)
            old_w = copy.deepcopy(self.w)

            self.w = self.w + r_k * p_k
            # 算法6.2——(6）
            self._cal_Gw()
            if self._cal_g_l2() < self.epsion:
                break
            y_k = self.g - old_g
            fai_k = self.w - old_w

            y_k = np.reshape(y_k,(self.n,1))
            fai_k = np.reshape(fai_k,(self.n,1))

            temp1 = np.dot(y_k,y_k.T) / float((np.dot(y_k.T,fai_k).reshape(1)[0]))
            temp2 = np.dot(np.dot(np.dot(self.B,fai_k),fai_k.T),self.B) / float(np.dot(np.dot(fai_k.T,self.B),fai_k).reshape(1)[0])
            self.B =self.B + temp1 - temp2
            

    def change_sample_feature_name(self,samples):
        new_samples = []
        for sample in samples:
            new_sample = []
            for feature_index,feature_v in enumerate(sample):
                new_feature_v = 'x' + str(feature_index) + '_' + str(feature_v)
                new_sample.append(new_feature_v)
            new_samples.append(np.array(new_sample))
        return np.array(new_samples)

    def _cal_Pxy_Px(self):
        # 从数据集中计算特征函数，f(x,y),有该样本就为1，没有则为0,x为样本X的某一个特征的取值
        self.P_x_y = defaultdict(int) # 其中P_x_y的键的个数则为特征函数的个数。
        self.P_x = defaultdict(int)

        for index in range(self.sample_num):
            # 取出样本值
            sample = self.samples[index]
            label = self.labels[index]

            for feature_index in range(self.feature_size):
                x = sample[feature_index]
                y = label
                self.P_x_y[(x,y)] = self.P_x_y[(x,y)] + 1
                self.P_x[x]  = self.P_x[x] + 1

    def _cal_EPxy(self):
        #计算特征函数f关于经验分布的P(x,y)的期望值
        self.EPxy = defaultdict(int) # 记录每个特征函数关于经验分布的P(x,y)的期望值
        #遍历特征函数，求出期望值
        for index in range(self.n):
            (x,y) = self.id2f[index]
            self.EPxy[index] = float(self.P_x_y[(x,y)]) / float(self.sample_num)

    def _f2id(self):
        #将index与特征函数对应起来
        self.id2f = {}
        self.f2id = {}
        for index,(x,y) in enumerate(self.P_x_y):
            self.id2f[index] = (x,y)
            self.f2id[(x,y)] = index

    def _cal_Pw(self,X,y):
        #《统计学习方法》公式6.28,计算Pw(y|x)，此处y只取0或1
        res = 0.
        for feature_v in X:
            if self.f2id.has_key((feature_v,y)):
                index = self.f2id[(feature_v,y)]
                res = res + (self.w[index] * 1)

        if y == 0:
            y = 1
        else:
            y = 0

        res_y = 0.
        for feature_v in X:
            if self.f2id.has_key((feature_v,y)):
                index = self.f2id[(feature_v,y)]
                res_y = res_y + (self.w[index] * 1)
        return float(res) / float(res + res_y)

    def _cal_Gw(self):

        # 计算f(w)对w_i的偏导数，《统计学习方法》P92,最上面g(w)的公式
        for index in range(self.n):
            res = 0.
            (x,y) = self.id2f[index]
            feature_index = int(x[1])
            # 累加
            for sample_index in range(self.sample_num):
                sample = self.samples[index]
                label = self.labels[index]

                if label != y:
                    continue
                if sample[feature_index] != x:
                    continue

                p_w = self._cal_Pw(sample, y)
                num = 0
                for feature_v in sample:
                    num = self.P_x[feature_v] + num
                #《统计学习方法》P82,计算P(X=x)公式
                p_x = float(num) / float(self.sample_num)
                res = res + p_w * p_x * 1 # 1为f_i特征函数的值

            self.g[index] = res - self.EPxy[index]

    def _cal_g_l2(self):
        res = sum(self.g * self.g) ** 0.5
        return res

    def _cal_fw(self):
        # 《统计学习方法》P91,f(w)计算公式
        res
        for index in range(self.n):
            (x,y) = self.id2f(index)


if __name__ == '__main__':
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    test_y = pd.read_csv(test_result_file)
    full_data = [train, test]

    # train.apply(axis=0)

    full_data = data_feature_engineering(full_data, age_default_avg=True, one_hot=False)
    train_X, train_y, test_X = data_feature_select(full_data)

    # lr = LR(iterNum=2000,learn_late=0.001)
    #
    # lr.train(train_X, train_y)
    #
    # results = []
    # for test_sample in test_X:
    #     sample = list(test_sample)
    #     sample.append(1.0)
    #     result = lr.predict(sample)
    #     results.append(result)
    #
    # y_test_true = np.array(test_y['Survived'])
    # print("the LR model Accuracy : %.4g" % metrics.accuracy_score(y_pred=results, y_true=y_test_true))

    mem = MEM()
    # 对于包含多特征属性的样本需要重新给每个属性值定义，用于区分f(x,y)中的x
    print(train_X[0:5])
    print('==============')
    print (mem.change_sample_feature_name(train_X[0:5]))
    # mem.train(train_X,train_y)