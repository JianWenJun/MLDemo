#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/5/16 上午11:44 
# @Author : ComeOnJian 
# @File : LR_MEM_demo.py 

import numpy as np
import pandas as pd
import random
import re
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
    def __init__(self):
        pass

    def train(self,train_X,train_y):
        # 使用《统计学习方法》P92算法6.2——BFGS，求解参数
        pass


if __name__ == '__main__':
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    test_y = pd.read_csv(test_result_file)
    full_data = [train, test]

    # train.apply(axis=0)

    full_data = data_feature_engineering(full_data, age_default_avg=True, one_hot=False)
    train_X, train_y, test_X = data_feature_select(full_data)

    lr = LR(iterNum=2000,learn_late=0.001)

    lr.train(train_X, train_y)

    results = []
    for test_sample in test_X:
        sample = list(test_sample)
        sample.append(1.0)
        result = lr.predict(sample)
        results.append(result)

    y_test_true = np.array(test_y['Survived'])
    print("the LR model Accuracy : %.4g" % metrics.accuracy_score(y_pred=results, y_true=y_test_true))
