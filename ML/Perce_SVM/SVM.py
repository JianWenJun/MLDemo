#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/5/10 下午5:14 
# @Author : ComeOnJian 
# @File : SVM.py

# 参考 SVM https://blog.csdn.net/sinat_33829806/article/details/78388025


import math
import numpy as np
import random
import copy
import re
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import svm

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
def Passenger_Survived(x):
    Survived = {0: -1, 1: 1}
    return Survived[x]
def get_title_name(name):
    title_s = re.search(' ([A-Za-z]+)\.', name)
    if title_s:
        return title_s.group(1)
    return ""

class SVM():
    def __init__(self,kernal,maxIter,C,epsilon,sigma = 0.001):
        """
        :param kernal:核函数
        :param maxIter:最大迭代次数
        :param C:松弛变量前的惩罚系数
        :param epseion:
        """
        self.kernal = kernal
        self.C = C
        self.maxIter = maxIter
        self.epsilon = epsilon
        self.sigma = sigma #高斯核函数的sigma值


    def train(self,train_X,train_y):

        self.sample_num = train_X.shape[0]
        self.feature_num = train_X.shape[1]

        self.labels =train_y
        self.samples = train_X

        # 算法的模型为 《统计学习方法》——公式7.104，主要包括a,b,核函数
        self.a = np.zeros(self.sample_num)#[0 for a_i in range(self.sample_num)]
        self.b = 0

        self.eCache = np.zeros(shape=(self.sample_num,2))# 存储差值
        self._smo()

        # self._update()

    def predict(self,test_x):
        # 《统计学习方法》——公式7.104，计算预测值
        pre_v = 0
        for index in range(self.sample_num):
            pre_v = pre_v + self.a[index] * self.labels[index] * self._kernel(test_x,self.samples[index])
        pre_v = pre_v + self.b
        return np.sign(pre_v)

    def _smo(self):
        pre_a = copy.deepcopy(self.a)  # 复制，pre_a是old的a数组
        for iter in range(self.maxIter):
            flag = 1
            for index in range(self.sample_num):
                diff = 0
                # self._update()
                E_i = self._calE(self.samples[index],self.labels[index])
                j,E_j = self._chooseJ(index,E_i)

                # 计算L H
                (L,H) = self._calLH(pre_a,j,index)

                # 《统计学习方法》——公式7.107，n = K11 + K22 - 2 * K12
                n = self._kernel(self.samples[index],self.samples[index]) \
                        + self._kernel(self.samples[j],self.samples[j])\
                        - 2 * self._kernel(self.samples[index],self.samples[j])
                if (n == 0):
                    continue
                # 《统计学习方法》——公式7.106，计算未剪切的a_j极值
                self.a[j] = pre_a[j] + float(self.labels[j] * (E_i - E_j))/n
                # 《统计学习方法》——公式7.108，计算剪切的a_j极值
                if self.a[j] > H:
                    self.a[j] = H
                elif self.a[j] < L:
                    self.a[j] = L
                # 《统计学习方法》——公式7.109，更新a[i]
                self.a[index] = pre_a[index] + self.labels[index] * self.labels[j] * (pre_a[j] - self.a[j])

                # 更新b,《统计学习方法》——公式7.114到7.116，更新a[i]
                b1 = self.b - E_i \
                     - self.labels[index] * self._kernel(self.samples[index],self.samples[index]) * (self.a[index] - pre_a[index]) \
                     - self.labels[j] * self._kernel(self.samples[j],self.samples[index]) * (self.a[j] - pre_a[j])
                b2 = self.b - E_j \
                     - self.labels[index] * self._kernel(self.samples[index], self.samples[j]) * (
                             self.a[index] - pre_a[index]) \
                     - self.labels[j] * self._kernel(self.samples[j], self.samples[j]) * (self.a[j] - pre_a[j])
                if (0 < self.a[index]< self.C):
                    self.b = b1
                elif (0 < self.a[j]< self.C):
                    self.b = b2
                else:
                    self.b = (b1 + b2)/2.0

                # 更新E_i,E_j统计学习方法》——公式7.117，
                self.eCache[j] = [1,self._calE(self.samples[j],self.labels[j])]
                self.eCache[index] = [1,self._calE(self.samples[index],self.labels[index])]

                diff = sum([abs(pre_a[m] - self.a[m]) for m in range(len(self.a))])
                if diff < self.epsilon:
                    # 满足精度条件
                    flag = 0
                pre_a = copy.deepcopy(self.a)

                if flag == 0:
                    break

    def _calE(self,sample,y):
        # 计算E_i,输入X_i与真实值之间的误差，《统计学习方法》——公式7.105
        pre_v = self.predict(sample)
        return pre_v - y

    def _calLH(self,a,j,i):
        #《统计学习方法》——p126页
        if(self.labels[j] != self.labels[i]):
            return (max(0,a[j]-a[i]),min(self.C,self.C+a[j]-a[i]))
        else:
            return (max(0, a[j] + a[i] - self.C), min(self.C, a[j] + a[i]))


    def _kernel(self,X_i,X_j):
        """
        :param X_i:
        :param X_j:
        :return: 核函数K(X_i,X_j)计算结果
        """
        result = 0.
        # 高斯内核
        if self.kernal == 'Gauss':
            temp = -sum((X_i - X_j)**2)/(2 * self.sigma**2)
            result = math.exp(temp)
        # 线性内核
        elif self.kernal == 'line':
            result = sum(X_i * X_j)

        return result

    def _chooseJ(self,i,E_i):
        # 选择变量
        self.eCache[i] = [1,E_i]
        choose_list = []
        # 查找之前计算的可选择的的E_i
        for cache_index in range(len(self.eCache)):
            if self.eCache[cache_index][0] != 0 and cache_index != i:
                choose_list.append(cache_index)
        if len(choose_list)>1:
            E_k =0
            delta_E = 0
            max_E = 0
            j = 0 # 要选择的J
            E_j = 0# 及其对应的E
            for choose_index in choose_list:
                E_k = self._calE(self.samples[choose_index],self.labels[choose_index])
                delta_E = abs(E_k-E_i)
                if delta_E > max_E:
                    max_E = delta_E
                    j = choose_index
                    E_j = E_k
            return j,E_j
        # 初始状态，没有已经计算好的E
        else:
            j = self._randJ(i)
            E_j = self._calE(self.samples[j],self.labels[j])
            return j , E_j

    def _randJ(self,i):
        j = i
        while(j == i):
            j = random.randint(0,self.sample_num-1)
        return j

if __name__ == '__main__':
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    test_y = pd.read_csv(test_result_file)
    train['Survived'] = train['Survived'].map(Passenger_Survived).astype(int)
    full_data = [train, test]

    full_data = data_feature_engineering(full_data, age_default_avg=True, one_hot=False)
    train_X, train_y, test_X = data_feature_select(full_data)

    svm1 = SVM('line',1000,0.05,0.001)
    svm1.train(train_X, train_y)
    results = []
    for test_sample in test_X:
        y = svm1.predict(test_sample)
        results.append(y)

    y_test_true = np.array(test_y['Survived'])
    print("the svm model Accuracy : %.4g" % metrics.accuracy_score(y_pred=results, y_true=y_test_true))

    # svm_s = svm.SVC(C=1,kernel='linear')
    # svm_s.fit(train_X, train_y)
    # pre_y = svm_s.predict(test_X)
    # y_test_true = np.array(test_y['Survived'])
    # print("the svm model Accuracy : %.4g" % metrics.accuracy_score(y_pred=pre_y, y_true=y_test_true))