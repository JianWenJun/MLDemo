#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/23 下午3:43 
# @Author : ComeOnJian 
# @File : decision_tree.py 

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle
from tqdm import tqdm
import copy


# ######################### 数据集操作 ################
##dataframe中的值数值化，包括一些简单的连续值离散化处理
def adult_label(x):
    """
    转换样本的类别
    :param s: key
    """
    label = {'>50K': 1,'<=50K' : 0,'>50K.': 1,'<=50K.' : 0}
    return label[x]
def adult_age(x):
    key = '17-35'
    x = int(x)
    if x>=36 and x<=53:
        key = '36-53'
    elif x>=54 and x<=71:
        key = '54-71'
    elif  x>=72 and x<=90:
        key = '72-90'
    age = {'17-35': 0, '36-53': 1, '54-71': 2, '72-90': 3}
    return age[key]
def adult_workclass(x):
    workclass = {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4,
                 'State-gov': 5, 'Without-pay': 6}
    return workclass[x]
def adult_education(x):
    education = {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5,
                 'Assoc-voc': 6, '9th': 7,
                 '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12, 'Doctorate': 13, '5th-6th': 14,
                 'Preschool': 15}
    return education[x]
def adult_education_num(x):
    key = '1-4'
    x = int(x)
    if x>=5 and x<=8:
        key = '5-8'
    elif x>=9 and x<=12:
        key = '9-12'
    elif  x>=13 and x<=16:
        key = '13-16'
    education_num = {'1-4': 0, '5-8': 1, '9-12': 2, '13-16': 3}
    return education_num[key]
def adult_marital_status(x):
    marital_status = {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4,
                      'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
    return marital_status[x]
def adult_occupation(x):
    occupation = {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4,
                  'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7,
                  'Adm-clerical': 8, 'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11,
                  'Protective-serv': 12, 'Armed-Forces': 13}
    return occupation[x]
def adult_relationship(x):
    relationship = {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative' :4, 'Unmarried': 5}
    return relationship[x]
def adult_race(x):
    race = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3,'Black': 4}
    return race[x]
def adult_sex(x):
    sex = {'Female': 0, 'Male': 1}
    return sex[x]
def adult_capital_gain_loss(x):
    capital_gain_loss = {'=0': 0, '>0': 1}
    key = '=0'
    x = int(x)
    if x>=0:
        key = '>0'
    return capital_gain_loss[key]
def adult_hours_per_week(x):
    hours_per_week = {'=40': 0, '>40': 1, '<40': 2}
    key = '=40'
    x = int(x)
    if x > 40:
        key = '>40'
    elif x < 40:
        key = '<40'
    return hours_per_week[key]
def adult_native_country(x):
    key = 'USA'
    native_country = {'USA': 0, 'not USA': 1}
    if x != 'United-States ':
        key = 'not USA'
    return native_country[key]

def transToValues(file_name,save_name,remove_unKnowValue=True,remove_duplicates=True):

    converters = {0: adult_age,1: adult_workclass,3: adult_education,4: adult_education_num,5: adult_marital_status,
                  6: adult_occupation,7: adult_relationship,8: adult_race,9: adult_sex,10: adult_capital_gain_loss,
                  11: adult_capital_gain_loss,12: adult_hours_per_week,13: adult_native_country,14: adult_label}
    adult_df = pd.read_table(file_name, header=None ,sep=',',converters=converters,
                                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                          'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                          'hours-per-week', 'native-country','label'],engine='python')
    if remove_duplicates:
        # 移除df中重复的值
        adult_df.drop_duplicates(inplace=True)
        print('delete duplicates shape train-test =================')
        print(adult_df.shape)

    if remove_unKnowValue:
        # 移除df中缺失的值
        adult_df.replace(['?'], np.NaN, inplace=True)
        adult_df.dropna(inplace=True)
        print('delete unKnowValues shape train-test =================')
        print(adult_df.shape)
        adult_df.drop('fnlwgt',axis=1,inplace=True)

    adult_df.to_csv(save_name,header=False,index=False)

if __name__ == '__main__':
    train_file = '../data/adult/adult.data'
    test_file = '../data/adult/adult.test'

    train_deal_file = '../data/adult/adult_deal.data'
    test_deal_file = '../data/adult/adult_deal.test'

    #deal with duplicates and unKnowValue
    transToValues(train_file,train_deal_file)
    transToValues(test_file,test_deal_file)

    # trans to value
    transToValues(train_deal_file,'../data/adult/adult_deal_value.data',remove_duplicates=False,remove_unKnowValue=False)
    transToValues(test_deal_file,'../data/adult/adult_deal_value.test',remove_duplicates=False,remove_unKnowValue=False)

def load_data(flods):
    adult_train_df = pd.read_table(flods[0], header=None ,sep=',',
                                   names=['age', 'workclass', 'education', 'education-num', 'marital-status',
                                          'occupation','relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                          'hours-per-week', 'native-country','label'],engine='python',dtype=int)
    adult_test_df = pd.read_table(flods[1], header=None, sep=',',
                                   names=['age', 'workclass', 'education', 'education-num', 'marital-status',
                                          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                          'hours-per-week', 'native-country', 'label'], engine='python',dtype=int)
    # 打乱data中样本的顺序
    # adult_train_df = shuffle(adult_train_df)
    adult_train_df = adult_train_df.sample(frac=1).reset_index(drop=True)
    # adult_test_df = shuffle(adult_test_df)
    adult_test_df = adult_test_df.sample(frac=1).reset_index(drop=True)
    print('init shape train-test =================')
    print(adult_train_df.shape)
    print(adult_test_df.shape)

    # 离散分段处理
    # D = np.array(adult_train_df['label']).reshape(adult_train_df.shape[0], 1)
    # age_did = devide_feature_value(adult_train_df['age'],D)

    train_data_x = np.array(adult_train_df.iloc[:,0:13])
    train_data_y = np.array(adult_train_df.iloc[:,13:])

    test_data_x = np.array(adult_test_df.iloc[:, 0:13])
    test_data_y = np.array(adult_test_df.iloc[:, 13:])

    return train_data_x,train_data_y,test_data_x,test_data_y

## 连续值离散处理
def devide_feature_value(series,D):
    sets = set(series)
    mid_value = []
    a = float(sets.pop())
    #取相邻点的中值
    for par in sets:
        a = (a + par) / 2.0
        mid_value.append(a)
        a = float(par)
    max_divide = mid_value[0]
    max_ent = 0.0
    ent_d = calc_ent(D)
    #查找最好的分裂点
    for mid in mid_value:
        Q1 = D[series < mid]
        Q2 = D[series >= mid]
        D_length = float(D.shape[0])
        Q1_length = Q1.shape[0]
        Q2_length = D_length - Q1_length
        #条件熵
        H_Q_D = Q1_length / D_length * calc_ent(Q1) + Q2_length / D_length * calc_ent(Q2)
        H = ent_d - H_Q_D
        if(H > max_ent):
            max_ent = H
            max_divide = mid
    return max_divide



# ######################### 数学计算 ################

def calc_ent(D):
    """
    计算数据集D的信息熵(经验熵),输入的labels
    :param x:数据集D,labels
    :return:ent
    """
    ent = 0.0

    x_value_list = set([D[i][0] for i in range(D.shape[0])]) #Ck 数据集中的类别数
    for x_value in x_value_list:
        p = float(D[D == x_value].shape[0]) / D.shape[0]
        logp = np.log2(p)
        ent -= p*logp
    return ent

def calc_condition_ent(A,D):
    """
    计算条件熵 H(D|A),x是特征项，y是D数据集labels
    :param x: A,某个特征项目
    :param y: D，数据集labels
    :return:条件熵
    """
    ent = 0.0
    x_value_list = set([A[i] for i in range(A.shape[0])]) #X 特征能取得值
    for x_value in x_value_list:
        sub_y = D[A == x_value] #Di
        ent_di = calc_ent(sub_y)
        ent += (float(sub_y.shape[0])/D.shape[0]) * ent_di
    return  ent

def calc_ent_gain(A,D):
    """
    :param A:某个特征项目
    :param D:数据集,labels
    :return:计算信息增益
    """
    ent_d = calc_ent(D)
    ent_condition_d_a = calc_condition_ent(A,D)
    return (ent_d - ent_condition_d_a)

def calc_ent_gain_rate(A,D):
    """
    计算信息增益比
    :param A:
    :param D:labels
    :return:信息增益比
    """
    ent = 0.0
    ent_gain = calc_ent_gain(A,D)

    x_values_list = set([A[i] for i in range(A.shape[0])])

    for x_value in x_values_list:
        sub_y = D[A == x_value]# Di
        p = float(sub_y.shape[0])/D.shape[0] #Di/D
        logp = np.log2(p)
        ent -= p * logp

    return ent_gain/ent

def calc_gini(D):
    """
    计算样本集D的基尼系数
    :param D: labels
    :return:
    """
    gini = 0.0
    x_value_list = set([D[i][0] for i in range(D.shape[0])])
    for x_value in x_value_list:
        Ck_count = D[D == x_value].shape[0]
        D_count = D.shape[0]
        p = float(Ck_count)/D_count
        gini += np.square(p)
    gini = 1 - gini
    return gini

def calc_condition_gini(A,D,a):
    """
    在特征A的条件下，集合D的基尼系数
    :param A:特征A
    :param D:labels
    :param a:特征A的确却的值a
    :return:
    """

    D1 = D[A == a]
    # D2 = D - D1 无此种形式
    # 取差集
    mask = A != a
    D2 = D[mask]
    p1 = float(D1.shape[0])/D.shape[0]
    p2 = float(D2.shape[0])/D.shape[0]

    gini1 = calc_gini(D1)
    gini2 = calc_gini(D2)

    gini = p1 * gini1 + p2 * gini2

    return gini

# ######################### 模型分类效果评价 ################

def eval(y_true,y_predict):

    from sklearn.metrics import average_precision_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import precision_score, recall_score, f1_score

    print('average_precision_score: %f' % (average_precision_score(y_true=y_true, y_score=y_predict)))
    print('MMC: %f' % (matthews_corrcoef(y_true=y_true, y_pred=y_predict)))

    print(classification_report(y_true=y_true, y_pred=y_predict))


    # Calculate precision score
    print(precision_score(y_true, y_predict, average='macro'))
    print(precision_score(y_true, y_predict, average='micro'))
    print(precision_score(y_true, y_predict, average=None))

    # Calculate recall score
    print(recall_score(y_true, y_predict, average='macro'))
    print(recall_score(y_true, y_predict, average='micro'))
    print(recall_score(y_true, y_predict, average=None))

    # Calculate f1 score
    print(f1_score(y_true, y_predict, average='macro'))
    print(f1_score(y_true, y_predict, average='micro'))
    print(f1_score(y_true, y_predict, average=None))

    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc))
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("Receiver Operating Characteristic, ROC(AUC = %0.2f)" % (roc_auc))
    plt.show()

# ######################### TreeNode ################

class TreeNode():
   """
   树节点类
   """
   def __init__(self):
       #叶子结点需要的属性

       self.type = -1  # 结点的类型label-类标记

       #非叶子结点需要的属性
       self.next_nodes = []  # 该结点指向的下一层结点
       self.feature_index = -1 #该结点对应的特征编号
       # self.feature_value = 0 #该结点划分的特征取值
       self.select_value = 0 #特征选择（信息增益、信息增益比、gini）值

   def add_next_node(self,node):
       if type(node) == TreeNode:
           self.next_nodes.append(node)
       else:
           raise Exception('node not belong to TreeNode type')
   def add_attr_and_value(self,attr_name,attr_value):
       """
       动态给节点添加属性，因为结点分为叶子结点，正常结点
       :param attr_name:属性名
       :param attr_value:属性值
       """
       setattr(self,attr_name,attr_value)

# ######################### Decision Tree  ################

class DecisionTree():
    def __init__(self,mode):
        self._tree = TreeNode() #指向根结点的指针
        if mode == 'ID3' or mode == 'C4.5':
            self._mode = mode
        else:
            raise Exception('mode should is C4.5 or ID3 or CARTClassification or CARTRegression')
    def train(self,train_x,train_y,epsoion):
        """
        构建树
        :param train_x:
        :param train_y:
        :return:该树 ———— 模型
        """
        feature_list = [index for index in range(train_x.shape[1])]

        self._create_tree(train_x,train_y,feature_list,epsoion,self._tree)
        # print (22)

    def predict(self,test_x):
        if (len(self._tree.next_nodes) == 0):
            raise Exception('no train model')

        # classfiy one sample
        def _classfiy(node,sample):
            feature_index = node.feature_index
            #叶子结点
            if feature_index == -1:
                return node.type
            #
            sample_feature_v = sample[feature_index]
            next_node = None
            for sub_node in node.next_nodes:
                if  hasattr(sub_node,'feature_value'):
                    if sub_node.feature_value == sample_feature_v:
                        next_node = sub_node
                        break;

            if next_node ==  None:
                return node.type
            else:
                return _classfiy(next_node,sample)


        predict_labels = []
        for sample in tqdm(test_x):
            label = _classfiy(self._tree.next_nodes[0],list(sample))
            predict_labels.append(label)
        return predict_labels

    def _create_tree(self,X,y,feature_list,epsoion,start_node,Vi=-1):
        """
        :param X: 数据集X
        :param y: label集合
        :param feature_list: 特征的id list
        :param epsoion:阈值
        :param start_node:决策树的启始结点
        :param Vi: feature value
        :return: 指向决策树的根结点的指针
        """
        # 结点
        node = TreeNode()
        #若所有实例都属于一个类别
        C = set(y[:,0]) #分类的类别集合
        if(len(C) == 1 ):
            node.type = tuple(C)[0] #该Ck作为该结点的类标记
            start_node.add_next_node(node)
            return

        # 特征集合A为空,将D中实例数最大的类Ck作为该结点的类标记
        if(len(feature_list) == 0):
            max_value = self._get_max_count_array(y[:,0])
            node.type = max_value
            start_node.add_next_node(node)
            return

        # select feature
        if self._mode == 'ID3' or self._mode == 'C4.5':
            select_func = calc_ent_gain
            if self._mode == 'C4.5':
                select_func = calc_ent_gain_rate
            ent_gain_max, ent_max_feature_index = self._select_feature(X,y,feature_list,select_func)
            # 最大信息增益小于设定的某个阈值
            if ent_gain_max < epsoion:
                type_value = self._get_max_count_array(y[:, 0])
                node.type = type_value
                start_node.add_next_node(node)
                return
            else:
                node.feature_index = ent_max_feature_index
                node.select_value = ent_gain_max
                type_value = self._get_max_count_array(y[:,0])
                node.type = type_value
                if (Vi != -1):
                    node.add_attr_and_value("feature_value", Vi)
                start_node.add_next_node(node)
                # 获取选取的特征的所有可能值
                Ag_v = set(X[:,ent_max_feature_index])
                # A - Ag
                feature_list.remove(ent_max_feature_index)
                # Di
                for v in Ag_v:
                    # Di 为 Xi , yi
                    mask = X[:,ent_max_feature_index] == v
                    Xi = X[mask]
                    yi = y[mask]
                    feature_list_new = copy.deepcopy(feature_list)
                    self._create_tree(Xi, yi, feature_list_new, epsoion, node, Vi=v)
                return
        else:
            pass

        pass

    def _select_feature(self,X,y,feature_list,select_func):
        ent_gain_max = 0.0
        ent_max_feature_index = 0
        for feature in feature_list:
            A = X[:,feature]
            D = y
            ent_gain = select_func(A,D)
            if(ent_gain > ent_gain_max):
                ent_gain_max = ent_gain
                ent_max_feature_index = feature

        return ent_gain_max,ent_max_feature_index


    def _get_max_count_array(self,arr):
        count = np.bincount(arr)
        max_value = np.argmax(count)
        return max_value


