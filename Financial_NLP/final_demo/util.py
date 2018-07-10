#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/6/15 下午8:54 
# @Author : ComeOnJian 
# @File : project.py 
# 参考https://github.com/YuriyGuts/pygoose/blob/master/pygoose
"""
整个项目的结构
"""
import os
import io
import numpy as np
import pandas as pd
import pickle


class Project:

    def __init__(self,root_dir):
        self._root_dir = root_dir
        self._init_all_paths()

    def _init_all_paths(self):
        self._data_dir = os.path.join(self._root_dir, 'data')#存放训练数据和测试数据
        # self._notebooks_dir = os.path.join(self._root_dir, 'notebooks')
        self._aux_data_dir = os.path.join(self._data_dir, 'external') #存放外部数据源
        self._preprocessed_data_dir = os.path.join(self._data_dir, 'preprocessed') ##存放预处理好的数据
        self._features_dir = os.path.join(self._data_dir, 'features') #存放抽取的特征数据
        # self._submissions_dir = os.path.join(self._data_dir, 'submissions') #存放最终提交的文件
        self._trained_model_dir = os.path.join(self._data_dir, 'trained') #存放训练的模型文件
        self._temp_dir = os.path.join(self._data_dir, 'tmp') #存放临时文件

    # 设置成只读属性
    @property
    def root_dir(self):
        return self._root_dir + os.path.sep

    @property
    def data_dir(self):
        return self._data_dir + os.path.sep

    @property
    def aux_dir(self):
        return self._aux_data_dir + os.path.sep

    @property
    def preprocessed_data_dir(self):
        return self._preprocessed_data_dir + os.path.sep

    @property
    def features_dir(self):
        return self._features_dir + os.path.sep

    @property
    def trained_model_dir(self):
        return self._trained_model_dir + os.path.sep

    @property
    def temp_dir(self):
        return self._temp_dir + os.path.sep

    # print os.getcwd()  # 获取当前工作目录路径
    # print os.path.abspath('.')  # 获取当前工作目录路径
    # print os.path.abspath('test.txt')  # 获取当前目录文件下的工作目录路径
    # print os.path.abspath('..')  # 获取当前工作的父目录 ！注意是父目录路径
    # print os.path.abspath(os.curdir)  # 获取当前工作目录路径

    @staticmethod
    def init(root_dir,create_dir = True):
        """
        :param root_dir:项目根目录
        :param create_dir:是否需要重新创建项目存放的资料目录
        :return:放回项目操作的对象
        """
        project = Project(root_dir)
        if create_dir:
            paths_to_create = [
                project.data_dir,
                project.aux_dir,
                project.features_dir,
                project.preprocessed_data_dir,
                project.trained_model_dir,
                project.temp_dir
            ]
            for path in paths_to_create:
                if os.path.exists(path):
                    continue
                else:
                    os.makedirs(path)
        return project

    """
    说明：
         某中方法提取特征后，产生多列的数据如下，分别存放两个文件中feature.name(列名)和feature.pickle(根据该抽取方法获得的样本数据)        
         列名:  f_1 f_2 f_3 f_4 f_5 f_6 f_7 f_8 f_9 f_10
         样本1: 0.1 0.4 8   2   4   2   3   0.1 0.4 0.33
         样本2: 0.1 0.4 8   2   4   2   3   0.1 0.4 0.33
         
    """
    def load_feature_lists(self,feature_lists):
        """
        根据特征名的列表，从运用各个方法抽取的特征文件中提出数据组合成DataFrame
        :param feature_lists:特征名组成的列表,可以将feature看成是抽取特征的方式
        :return:特征数据组成的DataFrame
        """
        column_names = []
        feature_ranges = []
        running_feature_count = 0

        # 从存放特征列名的文件中加载出列名，并记录各个特征对应的起始列的index
        for feature_name in feature_lists:
            feature_col_name_list = self._load_feature_col_name(self.features_dir + 'X_train_{}.names'.format(feature_name))
            column_names.extend(feature_col_name_list)
            start_index = running_feature_count
            end_index = running_feature_count + len(feature_col_name_list) - 1
            running_feature_count = running_feature_count + len(feature_col_name_list)
            feature_ranges.append([feature_name,start_index,end_index])

        # 从存放多个列文件中将数据的特征组合起来
        X_train = np.hstack([self._load_feature_data(self.features_dir + 'X_train_{}.pickle'.format(feature_name))
                                 for feature_name in feature_lists
                                 ])
        X_test = np.hstack([self._load_feature_data(self.features_dir + 'X_test_{}.pickle'.format(feature_name))
                                for feature_name in feature_lists
                                ])

        train_df = pd.DataFrame(X_train,columns=column_names)
        test_df = pd.DataFrame(X_test,columns=column_names)

        return train_df,test_df,feature_ranges

    def save_features(self,train_fea,test_fea,fea_names,feature_name):
        """
        使用某种方式使用本方法来保存特征
        :param train_fea:某种方法抽取的特征的多列数据来源于训练数据
        :param test_fea:某种方法抽取的特征的多列数据来源于测试数据
        :param fea_names:某种方法抽取的特征的形成的多列的数据对应的列名,list类型
        :param feature_name:抽取的方法名
        """
        self.save_feature_names(fea_names,feature_name)
        self.save_feature_col_list(train_fea,'train',feature_name)
        self.save_feature_col_list(test_fea,'test',feature_name)

    def save_feature_names(self,fea_names,feature_name):
        # 保存列名
        self._save_feature_col_name(fea_names,self.features_dir + 'X_train_{}.names'.format(feature_name))

    def save_feature_col_list(self,fea_data,type,feature_name):
        # 保存各列对应的数据
        self._save_feature_data(fea_data,self.features_dir + 'X_{}_{}.pickle'.format(type,feature_name))

    def _load_feature_col_name(self,nfile):

        with io.open(nfile,'r',encoding="utf-8") as file:
            return [line.rstrip('\n') for line in file.readlines()]

    def _load_feature_data(self,nfile):
        with open(nfile,'rb') as file:
            return pickle.load(file)

    def _save_feature_data(self,data,nfile):
        with open(nfile,'wb') as file:
            pickle.dump(data,file)

    def _save_feature_col_name(self,col_names,nfile):
        with open(nfile,'w') as file:
            file.write('\n'.join(col_names))

    def save(self,nfile,object):
        with open(nfile, 'wb') as file:
            pickle.dump(object, file)
    def load(self,nfile):
        with open(nfile, 'rb') as file:
            return pickle.load(file)

# 初始化整个项目的基础类
project = Project.init('/Users/jian/PythonPrMl/Financial_NLP/atec',create_dir=False)




