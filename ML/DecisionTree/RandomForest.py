#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/4/5 上午10:55 
# @Author : ComeOnJian 
# @File : RandomForst.py 

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import random
"""
随机示例对于回归问题此处采用的是平均法，对于分类问题采用的是投票法
"""
from enum import Enum

class TypeClass(Enum):
    DecisionTreeClassifier_type = 1
    DecisionTreeRegressor_type = 2

def randomforst(D,N,M,K,type_class):
    """
    :param D: 数据集D，格式为[Feature,label]，类型为np.ndarray
    :param N: 一次随机选取出的样本数
    :param M: M个基分类器
    :param k: 所有特征中选取K个属性
    :return:
    """
    D_df = pd.DataFrame(D)
    D_df.as_matrix()
    trees = []
    for count in M:
        # 随机采样N个样本
        D_df_i = D_df.sample(N)

        # 随机选取K个特征
        #包含label所以需要减1
        feature_length = D_df.shape[0] - 1
        feature_list = range(feature_length)
        choice_features = random.sample(feature_list,K)

        #最终的Di数据集
        D_df_i_data = D_df_i[choice_features]
        if isinstance(type_class,TypeClass):
            if type_class == TypeClass.DecisionTreeClassifier_type:
                cart_t = DecisionTreeClassifier(criterion='gini')
            else:
                cart_t = DecisionTreeRegressor(criterion='mse')
            y = D_df_i_data[-1].as_matrix()
            X = D_df_i_data.drop([-1], axis=1).as_matrix()
            tree = cart_t.fit(X, y)
            trees.append(tree)
        else:
            raise Exception('input param error')
        return trees
def randomforst_predict(trees,test_x, type_class):

    if isinstance(type_class, TypeClass):
        results = []
        for tree in trees:
            result = tree.predict(test_x)
            results.append(result)
        results_np = np.array(results)
        if type_class == TypeClass.DecisionTreeClassifier_type:
            return get_max_count_array(results_np)
        else:
            return np.mean(results_np)


    else:
        raise Exception('input param error')
def get_max_count_array(arr):
    count = np.bincount(arr)
    max_value = np.argmax(count)
    return max_value