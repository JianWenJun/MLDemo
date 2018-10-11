#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/6 上午10:42 
# @Author : ComeOnJian 
# @File : train_model.py 

import util
from gensim import corpora, models
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
# sklearn model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# file path
stop_words_path = './data/stopword.txt'
corpus_path = './data/model_data/corpus.mm'
corpora_dict_path = './data/model_data/dict.pickle'
corpus_docs_seg_path = './data/model_data/docs_seg.pickle'

# model path
tfidf_path = './data/model_data/tfidf_model.model'
lda_path = './data/model_data/lda_model.model'
lsi_path = './data/model_data/lsi_model.model'

# custom dict path
renming_dict_path = './data/mingxing.dict'
zhongyi_dict_path = './data/zyi.dict'
yinshi_dict_path = './data/yinshi.dict'
yaoping_dict_path = './data/yaoping.dict'

# data path
all_data_path = './data/all_docs.txt'
train_data_path = './data/train_1000.csv'
test_data_path = './data/test_107295.csv'
train_candidate_path = './data/train_1000_candidate_add_title.pickle'
test_candidate_path = './data/test_107295_candidate_add_title.pickle'
random_state = 4555

def get_topic_sim(model, word_corpus, doc_corpus):
    doc_topic_prob = model[doc_corpus]
    word_topic_prob = model[word_corpus]
    sim = util.cal_sim(word_topic_prob, doc_topic_prob)
    return  sim

def build_topic_model(data, stop_nfile, num_topics = 50, save = True):
    stop_words = util.stopwordslist(stop_nfile)
    corpora_documents = [] # 分词好的语料
    for index, row in data.iterrows():
        doc = str(row['doc']).strip()
        doc_seg = list(jieba.cut(doc))
        doc_seg_no_stop = [word for word in doc_seg if word not in stop_words]
        corpora_documents.append(doc_seg_no_stop)
        if index%3000 == 0:
            print('deal with sentence %d'%index)
    corpora_dict = corpora.Dictionary(corpora_documents)
    if save:
        util.save_object(corpora_documents, corpus_docs_seg_path)
        util.save_object(corpora_dict, corpora_dict_path)
    # corpora_documents = load_object('./data/docs_seg.pickle')
    # corpora_dict = load_object('./data/dict.pickle')

    corpus = [corpora_dict.doc2bow(doc) for doc in corpora_documents]
    # corpus每个元素为(word_id, fre)表示某个word在该doc中的fre词频
    # save corpus
    if save:
        corpora.MmCorpus.serialize(corpus_path,corpus)
    # load corpus
    # corpus = corpora.MmCorpus('./data/corpus.mm')

    # tf-idf model
    tfidf_model = models.TfidfModel(corpus)
    print('tf-idf model finish...')

    corpus_tfidf = tfidf_model[corpus]
    # lda model
    lda_model = models.LdaModel(corpus_tfidf, id2word=corpora_dict, num_topics=num_topics)
    print('lda model finish...')
    # lsi model
    # corpus_tfidf = tfidf_model[corpus]
    lsi_model = models.LsiModel(corpus_tfidf,id2word=corpora_dict,num_topics=num_topics)
    print('lsi model finish...')
    if save:
        tfidf_model.save(tfidf_path)
        lda_model.save(lda_path)
        lsi_model.save(lsi_path)

def bulid_candidate_words(data, stop_nfile, candidate_save_path, candidata_pos={}, first_sentence_count=30, last_sentence_count=20):
    # ID 标题 文本内容
    stop_words = util.stopwordslist(stop_nfile)
    # load corpus and model
    corpus_dict = util.load_object(corpora_dict_path)
    corpus = corpora.MmCorpus(corpus_path)
    tfidf_model = models.TfidfModel.load(tfidf_path)
    lda_model = models.LdaModel.load(lda_path)
    lsi_model = models.LsiModel.load(lsi_path)

    candidate_words = []
    for index, row in data.iterrows():
        title = str(row['title']).strip()
        doc = str(row['doc']).strip()
        candidate_word = {} # 该行记录的候选词key为word,value为id对应的特征(选择的10个特征)
        # doc
        words_doc = list(pseg.cut(doc, HMM=True)) #[(word, flag)]
        # title
        words_title = list(pseg.cut(title, HMM=True))

        # 去除停用词
        words_doc = [(word, pos) for word,pos in words_doc if word not in stop_words]
        words_title = [(word, pos) for word,pos in words_title if word not in stop_words]

        doc_len = len(words_doc)  # 统计去除停用词后的doc长度
        title_len = len(words_title)
        for word_index,(word,pos) in enumerate(words_doc):
            if pos in candidata_pos and len(word) > 1:
                # 特征的最后三项分别:features[-3]doc长度,features[-2]纪录候选词的首次出现位置,features[-1]最后一次出现的位置
                if word in candidate_word:
                    word_features = candidate_word[word]
                    word_features[-1] = (word_index+1)
                    candidate_word[word] = word_features
                    continue
                else:
                    features = [0] * 14
                    features[-3] = doc_len
                    # feature 1 词性
                    features[0] = candidata_pos[pos]
                    # feature 2 候选词首次出现的位置
                    if doc_len == 0:
                        firoc = 0.
                    else:
                        firoc = (word_index+1)/float(doc_len)
                    features[1] = firoc
                    features[-2] = (word_index+1) # 首次出现的位置
                    # feature 3 候选词的长度
                    features[2] = len(word)
                    # feature 4 候选词为的字符都是数字或者字母组成
                    if util.is_contain_char_num(word):
                        features[3] = 1
                    # feature 5 候选词对应的tfidf
                    id = corpus_dict.token2id.get(word, len(corpus_dict.token2id)+1)
                    if id == len(corpus_dict.token2id)+1:
                        features[4] = 1e-8
                    else:
                        for (w_id, tfidf) in tfidf_model[corpus[index]]:
                            if id == w_id:
                                features[4] = tfidf
                                break
                    # feature 6 第一句中候选词出现的次数
                    first_sentence = words_doc[:first_sentence_count]
                    features[5] = util.get_count_sentence(word,first_sentence)
                    # feature 7 最后一句中候选词出现的次数[-20:]
                    last_sentence = words_doc[-last_sentence_count:]
                    features[6] = util.get_count_sentence(word,last_sentence)
                    # feature 8,9 LDA,LSI:候选词的主题分布与文档的主题分布的相似度
                    single_list = [word]
                    word_corpus = tfidf_model[corpus_dict.doc2bow(single_list)]
                    features[7] = get_topic_sim(lda_model,word_corpus,corpus[index])
                    features[8] = get_topic_sim(lsi_model,word_corpus,corpus[index])
                    # feature 11 词跨度长度由的首次出现位置和最后一次出现的位置和doc长度计算
                    candidate_word[word] = features

        for word_index, (word, pos) in enumerate(words_title):
            if pos in candidata_pos and len(word) > 1:
                if word in candidate_word:
                    word_features = candidate_word[word]
                    # feature 10 是否出现在标题中
                    word_features[9] = 1
                    candidate_word[word] = word_features
                else:
                    features = [0] * 14
                    features[-3] = title_len
                    # feature 1 词性
                    features[0] = candidata_pos[pos]
                    # feature 2 候选词首次出现的位置
                    if title_len == 0:
                        firoc = 0.
                    else:
                        firoc = (word_index + 1) / float(title_len)
                    features[1] = firoc
                    features[-2] = (word_index + 1)  # 首次出现的位置
                    # feature 3 候选词的长度
                    features[2] = len(word)
                    # feature 4 候选词为的字符都是数字或者字母组成
                    if util.is_contain_char_num(word):
                        features[3] = 1
                    # feature 5 候选词对应的tfidf
                    id = corpus_dict.token2id.get(word, len(corpus_dict.token2id) + 1)
                    if id == len(corpus_dict.token2id) + 1:
                        features[4] = 1e-8
                    else:
                        for (w_id, tfidf) in tfidf_model[corpus[index]]:
                            if id == w_id:
                                features[4] = tfidf
                                break
                    # feature 6 第一句中候选词出现的次数
                    first_sentence = words_doc[:first_sentence_count]
                    features[5] = util.get_count_sentence(word, first_sentence)
                    # feature 7 最后一句中候选词出现的次数[-20:]
                    last_sentence = words_doc[-last_sentence_count:]
                    features[6] = util.get_count_sentence(word, last_sentence)
                    # feature 8,9 LDA,LSI:候选词的主题分布与文档的主题分布的相似度
                    single_list = [word]
                    word_corpus = tfidf_model[corpus_dict.doc2bow(single_list)]
                    features[7] = get_topic_sim(lda_model, word_corpus, corpus[index])
                    features[8] = get_topic_sim(lsi_model, word_corpus, corpus[index])
                    # feature 10 是否出现在标题中
                    features[9] = 1
                    # feature 11 词跨度长度由的首次出现位置和最后一次出现的位置和doc长度计算
                    candidate_word[word] = features
        candidate_words.append(candidate_word)
        # save
        if index % 2000 == 0:
            print('deal with sentence %d' % index)

    # data['candidate_words'] = candidate_words
    # data.to_csv(data_candidate_path, sep='\001', header=None, index=None)
    util.save_object(candidate_words,candidate_save_path)

def build_train_sample(data, candidate_words_list):
    """
    :param data: DataFrame
    :param candidate_words: list,[dict,dict,dict,...],其中dict
    """
    labels = []
    Featutes = []
    assert data.shape[0] == len(candidate_words_list)
    for index, row in data.iterrows():
        targets = str(row['key_words']).strip().split(',')
        candidate_words = candidate_words_list[index]
        if len(candidate_words) == 0:
            print('%d sentence is null'%(index))
        for word,feature in candidate_words.items():
            doc_len = feature[-3]
            first_index = feature[-2]
            last_index = feature[-1]
            if doc_len <= 0:
                print('%s of %d sentence doc len == 0' % (word, index))
                feature[-3] = 0.
            else:
                feature[-3] = (last_index - first_index) / float(doc_len)

            if None in feature:
                # feature[8] = 0.
                print('index: %d, word is %s feature has None, the None is %d' % (index,word,feature.index(None)))
                # continue
            Featutes += [feature[:-2]]
            # flag = False
            # for target in targets:
            #     if word == target:
            #         labels += [1]  # [[1]]
            #         flag = True
            #         break
            #     elif (word in target or target in word):
            #         labels += [1] # [[1]]
            #         flag = True
            #         break
            #
            # if flag is False:
            #     labels += [0] #[[0]]
            #
            # number is 1822, all is 2992
            if word in targets:
                labels += [1]
            else:
                labels += [0]
    return np.array(Featutes,dtype='float32'),np.array(labels)

def train_class_model(features,labels):

    X_train, X_test, y_train, y_test = train_test_split(features,labels,shuffle=True,random_state=random_state,test_size=0.1)
    print('x_train:',X_train.shape)
    print('y_train:',y_train.shape)
    dt = DecisionTreeClassifier(class_weight={0:1,1:10},max_depth=3)
    # dt = GaussianNB()
    # dt = RandomForestClassifier(class_weight={0: 1, 1: 10}, n_estimators=50, max_depth=3)
    # dt = LogisticRegression(class_weight={0:1,1:10})
    print('start train...')
    dt.fit(X_train,y_train)
    # dt.fit(features,labels)

    print('finish train...')
    y_pre = dt.predict(X_test)
    print('accuracy: ',accuracy_score(y_test,y_pre))
    print('f1 score: ',f1_score(y_test,y_pre))
    print('precision_score: ',precision_score(y_test,y_pre))
    print('recall_score: ',recall_score(y_test,y_pre))
    return dt

def get_test_sample_prob(model, test_data, test_candidates):
    assert test_data.shape[0] == len(test_candidates)
    labels = []
    for index, row in test_data.iterrows():
        # one sample
        sample_labes = {}
        if index==86399:
            print('test---')
        candidate_words = test_candidates[index]
        for word,feature in candidate_words.items():
            doc_len = feature[-3]
            first_index = feature[-2]
            last_index = feature[-1]
            if doc_len <=0:
                feature[-3] = 0.
            else:
                feature[-3] = (last_index - first_index) / float(doc_len)
            features =  np.array([feature[:-2]])
            y_pre = model.predict_proba(np.array(features))[0][1]
            sample_labes[word] = y_pre
        labels.append(sample_labes)

        if index % 2000 == 0:
            print('test with sentence %d' % index)
    return labels

if __name__ == '__main__':
    # load custom dict
    jieba.load_userdict(renming_dict_path)
    jieba.load_userdict(zhongyi_dict_path)
    jieba.load_userdict(yinshi_dict_path)
    jieba.load_userdict(yaoping_dict_path)

    # step1 训练topic相关的模型
    # ID 标题 文本内容
    all_data = pd.read_csv(all_data_path, sep='\001', header=None)
    all_data.columns = ['id', 'title', 'doc']
    build_topic_model(all_data,stop_words_path,num_topics=50)
    print('finish build topic model...')

    # step2
    # 抽取候选词
    train_data = pd.read_csv(train_data_path,sep='\001', header=None)
    train_data.columns = ['id', 'title', 'doc','key_words']
    bulid_candidate_words(train_data, stop_words_path,train_candidate_path, candidata_pos=util.pos_dict, first_sentence_count=30,
                          last_sentence_count=20)
    print('finish build train candidate word...')

    test_data = pd.read_csv(test_data_path, sep='\001', header=None)
    test_data.columns = ['id', 'title', 'doc']
    bulid_candidate_words(test_data, stop_words_path, test_candidate_path, candidata_pos=util.pos_dict,
                          first_sentence_count=30,
                          last_sentence_count=20)






