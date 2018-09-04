#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/8/28 下午4:03 
# @Author : ComeOnJian 
# @File : data_analy.py

import pandas as pd
import numpy as np
import pickle
import yaml
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from gensim.models import KeyedVectors
import torch
import torch.utils.data as torch_data


# text process

def get_stopwords(docs, min_df, max_d):
    """
    :param docs: (pd.Series) document
    :param min_df: 最小频率(int)
    :param max_df: 最大不超过(float,0.0<max_df<1.0)
    """
    vec = CountVectorizer(min_df=min_df,max_df=max_d)
    vec.fit_transform(docs)
    print('the stop word length is %d'%(len(vec.stop_words_)))
    return vec.stop_words_

def make_vocab(data_se, stop_words, max_size, type, isSave=True):
    """
    :param data_se: the list of documents
    :param max_size: the vocab max size
    :param type: 1 is article ,2 is word_seg
    """
    # count the word which not in stop_words
    dict_fre_word = defaultdict(int)
    dict_id2label_word = {}
    dict_label2id_word = {}
    lengths = []
    word_index = 1  # 从1开始index
    for index in range(len(data_se)):
        #  split
        art_splits = data_se[index].split()
        lengths.append(len(art_splits))
        # count
        for word in art_splits:
            if word not in stop_words:
                dict_fre_word[word] += 1
                if word not in dict_label2id_word:
                    dict_id2label_word[word_index] = word
                    dict_label2id_word[word] = word_index
                    word_index += 1

    # save
    def save_dict(new_dicts,size, type, issave=True):
        if issave:
            if type == 1:
                save('./data/dicts_{}_arts'.format(size), new_dicts)
            if type == 2:
                save('./data/dicts_{}_words'.format(size), new_dicts)

    print('start prune vocab...')
    # prune to max size
    if len(dict_fre_word) <= max_size:
        new_dicts = {
            'dict_label2id': dict_label2id_word,
            'dict_id2label': dict_id2label_word,
            'stop_word': stop_words
        }
        save_dict(new_dicts,len(dict_fre_word),type,isSave)
        print('after prune the dicts is %d'%len(dict_fre_word))
    else:
        fres = np.array(list(dict_fre_word.values()))
        words = np.array(list(dict_fre_word.keys()))
        dict_id2label_new = {}
        dict_label2id_new = {}
        word_index_new = 1  # 从1开始index

        fres_index = np.argsort(-fres)  # 降序排列
        for index in fres_index[:max_size]:
            word = words[index]
            if word not in dict_label2id_new:
                dict_label2id_new[word] = word_index_new
                dict_id2label_new[word_index_new] = word
                word_index_new += 1

        new_dicts = {
            'dict_label2id': dict_label2id_new,
            'dict_id2label': dict_id2label_new,
            'stop_word': stop_words
        }
        save_dict(new_dicts, max_size, type, isSave)
        print('after prune the dicts is %d' % max_size)

def save(nfile, obj):
    with open(nfile,'wb') as p:
        pickle.dump(obj,p)

def load(nfile):
    with open(nfile,'rb') as r:
        a = pickle.load(r)
        return a

def init_vocab(min_df, max_d, add_test=True, char_vocab_size = 10000, word_vocab_size=200000):
    train = pd.read_csv('train_set.csv')
    all_article = train['article'].tolist()
    all_word_seg = train['word_seg'].tolist()
    if add_test:
        test = pd.read_csv('test_set.csv')
        test_article = test['article'].tolist()
        test_word_seg = test['word_seg'].tolist()
        all_article = all_article + test_article
        all_word_seg = all_word_seg + test_word_seg
    # article
    print('get the article stop_words, the word pro > %f or word count < %d'%(max_d,min_df))
    article_stop_words = get_stopwords(all_article,min_df=min_df,max_d=max_d)
    print('start init article vocab...')
    make_vocab(all_article, article_stop_words, char_vocab_size, type=1, isSave=True)

    # word_seg
    print('get the word_seg stop_words, the word pro > %f or word count < %d' % (max_d, min_df))
    word_stop_words = get_stopwords(all_word_seg, min_df=min_df, max_d=max_d)
    print('start init article vocab...')
    make_vocab(all_word_seg, word_stop_words, word_vocab_size, type=2, isSave=True)

def pre_train_w2v(all_text):


    model = word2vec.Word2Vec(sentences=texts, size=300, window=2, min_count=3, workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + "train_all_data.bigram", binary=binary, fvocab=None)

# text to index and save
def sentence_to_indexs(sentences, dict_label2id, stop_words, max_document_length=1500, print_count=3000, padding=True):
    all_sentence_indexs = []
    dict_size = len(dict_label2id)
    print('dict_size',dict_size)
    for index,sentence in enumerate(sentences):
        words = sentence.split()
        sentence_indexs = []
        for word in words:
            if word not in stop_words:
                if word in dict_label2id:
                    word_index = dict_label2id[word]
                else:
                    word_index = (dict_size + 1) # unknow word
                sentence_indexs.append(word_index)
            else:
                word_index = (dict_size + 2) # stop words
                sentence_indexs.append(word_index)
            if padding:
                words_length = len(sentence_indexs)
                if words_length < max_document_length:
                    # padding
                    for _ in range(max_document_length-words_length):
                        sentence_indexs.append(dict_size + 2) # padding word
                else:
                    sentence_indexs = sentence_indexs[:max_document_length] # prune sentence

        if len(sentence_indexs)==0:
            print(index)
        all_sentence_indexs.append(sentence_indexs)

        if (index+1) % print_count == 0:
            print('already deal with %d documents'%(index+1))
    return all_sentence_indexs

def split_train_val(data, article_dicts, word_dicts, rate=0.7, isSave = True):
    # data = pd.read_csv('train_set.csv')
    # article_dicts = load('./data/art_dicts')
    # word_dicts = load('./data/art_dicts')
    article_words = data[['article','word_seg']].values
    y = data['class'].values
    x_train, x_test, y_train, y_test = train_test_split(article_words,y,test_size=(1-rate),random_state=42)

    x_train_articles = x_train[:,0]
    x_train_words = x_train[:,1]
    x_test_article = x_test[:,0]
    x_test_words = x_test[:,1]

    x_train_articles_ids = sentence_to_indexs(x_train_articles,article_dicts['dict_label2id'],article_dicts['stop_word'],padding=False)
    x_train_words_ids = sentence_to_indexs(x_train_words,word_dicts['dict_label2id'],word_dicts['stop_word'],padding=False)
    x_test_articles_ids = sentence_to_indexs(x_test_article,article_dicts['dict_label2id'],article_dicts['stop_word'],padding=False)
    x_test_words_ids = sentence_to_indexs(x_test_words,word_dicts['dict_label2id'],word_dicts['stop_word'],padding=False)
    if isSave:
        save('./data/x_train_articles_ids.pickle',x_train_articles_ids)
        save('./data/x_train_words_ids.pickle',x_train_words_ids)
        save('./data/x_test_articles_ids.pickle',x_test_articles_ids)
        save('./data/x_test_words_ids.pickle',x_test_words_ids)
        save('./data/y_train.pickle',y_train)
        save('./data/y_test.pickle',y_test)
    return x_train_articles_ids, x_train_words_ids, x_test_articles_ids, x_test_words_ids, y_train, y_test

############ Constructing  model data set ###################

class dataset(torch_data.Dataset):

    def __init__(self, src_article, src_word, y):

        self.src_article = src_article
        self.src_word = src_word
        self.y = y

    def __getitem__(self, index):
        return self.src_article[index], self.src_word[index], self.y[index]

    def __len__(self):
        return len(self.src_article)

def padding(data):
    src_article, src_word, y = zip(*data)
    article_max_length = 2000
    word_seg_max_length = 1500
    # article
    art_src_len = [len(article) for article in src_article]
    if max(art_src_len) < article_max_length:
        article_max_length = max(art_src_len)
    art_src_pad = torch.zeros(len(src_article), article_max_length).long()
    for i, s in enumerate(src_article):
        if len(s) == 0 or s is None:
            print('None-------')
        end = art_src_len[i]
        # print(end)/
        if end > article_max_length:
            end = article_max_length
        s = torch.LongTensor(s)
        art_src_pad[i, :end] = s[:end]

    # word_seg
    word_src_len = [len(word) for word in src_word]
    if max(word_src_len) < word_seg_max_length:
        word_seg_max_length = max(word_src_len)
    word_src_pad = torch.zeros(len(src_word), word_seg_max_length).long()
    for i, s in enumerate(src_word):
        end = word_src_len[i]
        if end > word_seg_max_length:
            end = word_seg_max_length
        s = torch.LongTensor(s)
        word_src_pad[i, :end] = s[:end]
    # print(y[0],type(y[0]))

    y = [int(item)-1 for item in y]  # lable -> [0,18]

    return torch.LongTensor(art_src_pad), torch.LongTensor(word_src_pad), torch.LongTensor(y)

def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

############ analy config file ###################

class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # self.__dict__ = self

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setstate__(self, state):
        pass
    def __getstate__(self):
        pass
        # for key in state.keys():
        #     print(key,state[key])
        #     setattr(self,key,state[key])

def read_config(path):

    return AttrDict(yaml.load(open(path, 'r')))

if __name__ == '__main__':
    # step1 init vocab
    # char_vocab_size = 10000
    # word_vocab_size =  250000
    # init_vocab(min_df=3,max_d=0.9,add_test=True, char_vocab_size=char_vocab_size, word_vocab_size=word_vocab_size)
    # pre_train_w2v
    #

    # step2 sentece to ids
    # train = pd.read_csv('train_set.csv')
    # article_dicts = load('./data/dicts_10000_arts')
    # word_dicts = load('./data/dicts_250000_words')
    # x_train_articles_ids, x_train_words_ids, x_test_articles_ids, x_test_words_ids, y_train, y_test = split_train_val(train, article_dicts, word_dicts, rate=0.7, isSave=True)
    #
    #
    pass



