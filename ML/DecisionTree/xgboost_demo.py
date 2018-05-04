#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/4/28 下午7:24 
# @Author : ComeOnJian 
# @File : xgboost.py 

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt

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

def modelfit(alg,dtrain_x,dtrain_y,useTrainCV=True,cv_flods=5,early_stopping_rounds=50):
    """
    :param alg: 初始模型
    :param dtrain_x:训练数据X
    :param dtrain_y:训练数据y（label）
    :param useTrainCV: 是否使用cv函数来确定最佳n_estimators
    :param cv_flods:交叉验证的cv数
    :param early_stopping_rounds:在该数迭代次数之前，eval_metric都没有提升的话则停止
    """
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain_x,dtrain_y)
        cv_result = xgb.cv(xgb_param,xgtrain,num_boost_round = alg.get_params()['n_estimators'],
                           nfold = cv_flods,metrics = 'auc',early_stopping_rounds=early_stopping_rounds)

        # print(cv_result)
        alg.set_params(n_estimators=cv_result.shape[0])

    # train data
    alg.fit(train_X,train_y,eval_metric='auc')

    #predict train data
    train_y_pre = alg.predict(train_X)

    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_y_pre))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind = 'bar',title='Feature Importance')
    plt.ylabel('Feature Importance Score')
    plt.show()
    return alg
def xgboost_change_param(train_X,train_y):

    # Xgboost 调参

    # step1 确定学习速率和迭代次数n_estimators，即集分类器的数量
    xgb1 = XGBClassifier(learning_rate=0.1,
                         booster='gbtree',
                                   n_estimators=300,
                                   max_depth=4,
                                   min_child_weight=1,
                                   gamma=0,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   objective='binary:logistic',
                                   nthread=2,
                                   scale_pos_weight=1,
                                   seed=10
                                   )
    #最佳 n_estimators = 59 ，learning_rate=0.1
    modelfit(xgb1,train_X,train_y,early_stopping_rounds=45)

    # setp2 调试的参数是min_child_weight以及max_depth
    param_test1 = {
        'max_depth': range(3,8,1),
        'min_child_weight':range(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,n_estimators=59,
                                                    max_depth=4,min_child_weight=1,gamma=0,
                                                    subsample=0.8,colsample_bytree=0.8,
                                                    objective='binary:logistic',nthread=2,
                                                    scale_pos_weight=1,seed=10
                                                    ),
                            param_grid=param_test1,
                            scoring='roc_auc',n_jobs=1,cv=5)
    gsearch1.fit(train_X,train_y)
    print gsearch1.best_params_,gsearch1.best_score_
    # 最佳 max_depth = 7 ，min_child_weight=3
    # modelfit(gsearch1.best_estimator_) 最佳模型为：gsearch1.best_estimator_

    # step3 gamma参数调优
    param_test2 = {
        'gamma': [i/10.0 for i in range(0,5)]
    }
    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,n_estimators=59,
                                                    max_depth=7,min_child_weight=3,gamma=0,
                                                    subsample=0.8,colsample_bytree=0.8,
                                                    objective='binary:logistic',nthread=2,
                                                    scale_pos_weight=1,seed=10),
                            param_grid=param_test2,
                            scoring='roc_auc',
                            cv=5
                            )
    gsearch2.fit(train_X, train_y)
    print gsearch2.best_params_, gsearch2.best_score_
    # 最佳 gamma=0.3
    # modelfit(gsearch2.best_estimator_)

    #step4 调整subsample 和 colsample_bytree 参数
    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,n_estimators=59,
                                                    max_depth=7,min_child_weight=3,gamma=0.3,
                                                    subsample=0.8,colsample_bytree=0.8,
                                                    objective='binary:logistic',nthread=2,
                                                    scale_pos_weight=1,seed=10),
                            param_grid=param_test3,
                            scoring='roc_auc',
                            cv=5
                            )
    gsearch3.fit(train_X, train_y)
    print gsearch3.best_params_, gsearch3.best_score_
    # 最佳'subsample': 0.8, 'colsample_bytree': 0.6

    # step5 正则化参数调优

if __name__ == '__main__':
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    test_y = pd.read_csv(test_result_file)

    full_data = [train,test]

    # train.apply(axis=0)

    full_data = data_feature_engineering(full_data,age_default_avg=True,one_hot=False)
    train_X, train_y, test_X = data_feature_select(full_data)

    # XGBoost调参
    # xgboost_change_param(train_X,train_y)

    xgb = XGBClassifier(learning_rate=0.1,n_estimators=59,
                        max_depth=7,min_child_weight=3,
                        gamma=0.3,subsample=0.8,
                        colsample_bytree=0.6,objective='binary:logistic',
                        nthread=2,scale_pos_weight=1,seed=10)
    xgb.fit(train_X,train_y)

    y_test_pre = xgb.predict(test_X)
    y_test_true = np.array(test_y['Survived'])
    print ("the xgboost model Accuracy : %.4g" % metrics.accuracy_score(y_pred=y_test_pre, y_true=y_test_true))
