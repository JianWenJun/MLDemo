#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/3/7 下午4:06 
# @Author : ComeOnJian
# @File : text_cnn.py
# implementation of Convolutional Neural Networks for Sentence CLassification

import argparse
import pandas as pd
# import ast
import numpy as np
# import NLP.Text_CNN.process_data as process_data
# import NLP.Text_CNN.text_cnn_model as TextCNN
import process_data
from text_cnn_model import TextCNN
import pickle
from tqdm import tqdm

import pdb
# step1 get paramater
# step2 load data
# step3 create TextCNN model
# step4 start train
# step5 validataion

if __name__ == '__main__':
    # step1 get paramater
    parse = argparse.ArgumentParser(description='Paramaters for construct TextCNN Model')
    # #方式一 type = bool
    # parse.add_argument('--nonstatic',type=ast.literal_eval,help='use textcnn nonstatic or not',dest='tt')
    # 方式二 取bool值的方式)添加互斥的参数
    group_static = parse.add_mutually_exclusive_group(required=True)
    group_static.add_argument('--static', dest='static_flag', action='store_true', help='use static Text_CNN')
    group_static.add_argument('--nonstatic', dest='static_flag', action='store_false', help='use nonstatic Text_CNN')

    group_word_vec = parse.add_mutually_exclusive_group(required=True)
    group_word_vec.add_argument('--word2vec', dest='wordvec_flag', action='store_true', help='word_vec is word2vec')
    group_word_vec.add_argument('--rand', dest='wordvec_flag', action='store_false', help='word_vec is rand')

    group_shuffer_batch = parse.add_mutually_exclusive_group(required=False)
    group_shuffer_batch.add_argument('--shuffer', dest='shuffer_flag', action='store_true', help='the train do shuffer')
    group_shuffer_batch.add_argument('--no-shuffer', dest='shuffer_flag', action='store_false',
                                     help='the train do not shuffer')

    parse.add_argument('--learnrate', type=float, dest='learnrate', help='the NN learnRate', default=0.05)
    parse.add_argument('--epochs', type=int, dest='epochs', help='the model train epochs', default=10)
    parse.add_argument('--batch_size', type=int, dest='batch_size', help='the train gd batch size.(50-300)', default=50)
    parse.add_argument('--dropout_pro', type=float, dest='dropout_pro', help='the nn layer dropout_pro', default=0.5)

    parse.set_defaults(static_flag=True)
    parse.set_defaults(wordvec_flag=True)
    parse.set_defaults(shuffer_flag=False)

    args = parse.parse_args()


    # step2 load data
    print('load data. . .')
    X = pickle.load(open('./NLP/result/word_vec.p','rb'))

    word_vecs_rand, word_vecs, word_cab, sentence_max_len, revs = X[0],X[1],X[2],X[3],X[4]

    print('load data finish. . .')
    # configuration tf
    filter_sizes = [3, 4, 5]
    filter_numbers = 100
    embedding_size = 300
    # use word2vec or not
    W = word_vecs_rand
    if args.wordvec_flag:
        W = word_vecs
        pass
    # pdb.set_trace()
    word_ids,W_list = process_data.getWordsVect(W)

    # use static train or not
    static_falg = args.static_flag
    # use shuffer the data or not
    shuffer_falg = args.shuffer_flag
    #交叉验证
    results = []
    for index in tqdm(range(10)):
        #打调试断点
        # pdb.set_trace()
        # train_x, train_y, test_x, test_y = process_data.get_train_test_data1(W,revs,index,sentence_max_len,default_values=0.0,vec_size=300)
        train_x, train_y, test_x, test_y = process_data.get_train_test_data2(word_ids,revs,index,sentence_max_len)
        # step3 create TextCNN model
        text_cnn = TextCNN(W_list,shuffer_falg,static_falg,filter_numbers,filter_sizes,sentence_max_len,embedding_size,args.learnrate,args.epochs,args.batch_size,args.dropout_pro)
        # step4 start train
        text_cnn.train(train_x,train_y)
        # step5 validataion
        accur,loss = text_cnn.validataion(test_x, test_y)
        #
        results.append(accur)
        print('cv {} accur is :{:.3f} loss is {:.3f}'.format(index+1,accur,loss))
        text_cnn.close()
    print('last accuracy is {}'.format(np.mean(results)))



