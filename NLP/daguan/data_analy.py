#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/8/28 下午4:03 
# @Author : ComeOnJian 
# @File : data_analy.py

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec
from gensim.models import KeyedVectors


# text process
def make_vocab(data_se):
    """
    :param data_se: list
    """
    dict_fre_word = defaultdict(int)
    dict_id2label_word = {}
    dict_label2id_word = {}
    lengths = []
    word_index = 0
    for index in range(len(data_se)):
        #  split
        art_splits = data_se[index].split()
        lengths.append(len(art_splits))
        # count
        for word in art_splits:
            dict_fre_word[word] += 1
            if word not in dict_label2id_word:
                dict_id2label_word[word_index] = word
                dict_label2id_word[word] = word_index
                word_index += 1

    return dict_fre_word, dict_id2label_word, dict_label2id_word, lengths

def prune(data, docs, min_df, max_d):
    """
    :param docs: (pd.Series) document
    :param min_df: 最小频率(int)
    :param max_df: 最大不超过(float,0.0<max_df<1.0)
    """
    dict_fre_word, dict_id2label_word, dict_label2id_word = data[0], data[1], data[2]
    print('before prune dict length is %d ...'%(len(dict_fre_word)))
    vec = CountVectorizer(min_df=min_df,max_df=max_d)
    vec.fit_transform(docs)
    for word in vec.stop_words_:
        id = dict_label2id_word[word]
        dict_fre_word.pop(word)
        dict_label2id_word.pop(word)
        dict_id2label_word.pop(id)

    print('after prune dict length is %d, stop words length is %d...' % (len(dict_fre_word),len(vec.stop_words_)))
    return dict_fre_word, dict_id2label_word, dict_label2id_word, vec.stop_words_

def prune_to_max_size(nfile,max_size,type=1):
    """
    :param nfile: the dics file name
    :param max_size: the vocab max size
    :param type: 1 is article ,2 is word_seg
    """
    dicts = load(nfile)
    if type == 1:
        dict_fre = dicts['dict_fre_art']
        dict_id2label = dicts['dict_id2label_art']
        dict_label2id = dicts['dict_label2id_art']
    if type == 2:
        dict_fre = dicts['dict_fre_word_seg']
        dict_id2label = dicts['dict_id2label_word_seg']
        dict_label2id = dicts['dict_label2id_word_seg']

    if len(dict_fre) <= max_size:
        return None
    # Only keep the `size` most frequent entries.
    else:
        fres = np.array(list(dict_fre.values()))
        words = np.array(list(dict_fre.keys()))

        fres_index = np.argsort(-fres) # 降序排列
        for index in fres_index[:max_size]:
            word = words[index]
            id = dict_id2label[word]
            dict_fre.pop(word)
            dict_label2id.pop(word)
            dict_id2label.pop(id)

    return dict_fre, dict_label2id, dict_id2label

def save(nfile, obj):
    with open(nfile,'wb') as p:
        pickle.dump(obj,p)

def load(nfile):
    with open(nfile,'rb') as r:
        a = pickle.load(r)
        return a

def init_vocab(add_test=True):
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
    print('start init article vocab...')
    dict_fre_art, dict_id2label_art, dict_label2id_art, art_lengths = make_vocab(all_article)
    print('finish init article vocab and dict length is  %d - %d - %d'%(len(dict_fre_art),len(dict_id2label_art),len(dict_label2id_art)))
    print('start prune article vocab...')
    art_data = (dict_fre_art, dict_id2label_art, dict_label2id_art)
    dict_fre_art, dict_id2label_art, dict_label2id_art, stop_word_art = prune(art_data,all_article,min_df=3,max_d=0.9)

    # word_seg
    print('start init word_seg vocab...')
    dict_fre_word_seg, dict_id2label_word_seg, dict_label2id_word_seg, word_seg_lengths = make_vocab(all_word_seg)
    print('finish init word_seg vocab and dict length is  %d - %d - %d'%(len(dict_fre_word_seg),len(dict_id2label_word_seg),len(dict_label2id_word_seg)))
    print('start prune word_seg vocab...')
    word_data = (dict_fre_word_seg, dict_id2label_word_seg, dict_label2id_word_seg)
    dict_fre_word_seg, dict_id2label_word_seg, dict_label2id_word_seg, stop_word_word_seg = prune(word_data,all_word_seg,min_df=3,max_d=0.9)

    # save data
    art_dicts ={
        'dict_fre_art': dict_fre_art,
        'dict_id2label_art': dict_id2label_art,
        'dict_label2id_art': dict_label2id_art,
        'stop_word_art': stop_word_art,
        'art_lengths': art_lengths
    }
    word_dicts ={
        'dict_fre_word_seg': dict_fre_word_seg,
        'dict_id2label_word_seg': dict_id2label_word_seg,
        'dict_label2id_word_seg': dict_label2id_word_seg,
        'stop_word_word_seg': stop_word_word_seg,
        'word_seg_lengths': word_seg_lengths
    }
    save('./data/art_dicts',art_dicts)
    save('./data/word_dicts',word_dicts)

def pre_train_w2v(all_text):


    model = word2vec.Word2Vec(sentences=texts, size=300, window=2, min_count=3, workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + "train_all_data.bigram", binary=binary, fvocab=None)

# text to index and save
def sentence_to_indexs(sentences, dict_label2id, stop_words, max_document_length=1500, print_count=3000, padding=True):
    all_sentence_indexs = []
    dict_size = len(dict_label2id)
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
            if padding:
                words_length = len(sentence_indexs)
                if words_length < max_document_length:
                    # padding
                    for _ in range(max_document_length-words_length):
                        sentence_indexs.append(dict_size + 2) # padding word
                else:
                    sentence_indexs = sentence_indexs[:max_document_length] # prune sentence
        all_sentence_indexs.append(sentence_indexs)

        if (index+1) % print_count == 0:
            print('already deal with %d documents'%(index+1))

def split_train_val(data,rate=0.7):

    pass



if __name__ == '__main__':
    # init vocab
    # init_vocab(add_test=True)
    # pre_w2v
    # pre_train_w2v
    pass
