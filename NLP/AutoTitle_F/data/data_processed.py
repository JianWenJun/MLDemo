#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/24 下午4:51 
# @Author : ComeOnJian 
# @File : data_processed.py
from stanfordcorenlp import StanfordCoreNLP
import nltk
import re
import sys
sys.path.append('../')
import util
from collections import Counter

from string import punctuation

# nlp_tokenizer = StanfordCoreNLP('/home/jwj/pythonPro/stanford-corenlp-full-2018-02-27/', lang='en')
nlp_tokenizer = StanfordCoreNLP('/home/mlc/pythonPro/stanford-corenlp-full-2018-02-27/', lang='en')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
add_punc = '，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=擅长于的&#@￥'
all_punc = punctuation+add_punc

#################### 文本的清洗 ###########################

def transform_other_word(str_text,reg_dict):
    for token_str,replac_str in reg_dict.items():
        str_text = str_text.replace(token_str, replac_str)
    return str_text

def clean_text(text, contractions):
    text = transform_other_word(text, contractions)
    text = re.sub('&amp;', '', text)
    text = re.sub('[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub('\'', ' ', text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    return text

def pre_word_token(df, config, test = False, lower = True, is_make_title = False):
    """
    :param df: 数据
    :param config:配置信息
    :return:
    """
    if test:
        for index, row in df.iterrows():
            content = str(row['content']).strip("'<>() ")
            if lower:
                content = content.lower()
            content_words = word_tokenizer(content)
            df.at[index, 'content'] = " ".join(content_words)
            # 虚构title
            if is_make_title:
                df.at[index, 'title'] = "test"
        df.to_csv('./results/test_set_add_title.csv', sep='\t', header=None, index=None, encoding='utf-8')
    else:
        if config.makevocab:
            vocab_counter = Counter()
        delete_ids = []
        # 去重
        df.drop_duplicates(['title'], inplace=True)
        df.drop_duplicates(['content'], inplace=True)
        for index, row in df.iterrows():
            try:
                title = str(row['title']).strip("'<>() ")
                content = str(row['content']).strip("'<>() ")
                if lower:
                    title = title.lower()
                    content = content.lower()
                title_words = word_tokenizer(title)
                content_words = word_tokenizer(content)

                if len(title_words) == 1:
                    delete_ids.append(index)
                    print('ignore id: {} sentence...,due to title length is 1'.format(row['id']))
                    continue
                df.at[index,'title'] = " ".join(title_words)
                df.at[index, 'content'] = " ".join(content_words)
                ### bulid Vocab
                if config.makevocab:
                    vocab_counter.update(title_words)
                    vocab_counter.update(content_words)
                if index % 5000 == 0:
                    print('has deal with %d sentence...'% index)
            except Exception:
                print('id :{} sentence can not deal with, because the sentence length is {} exceed corenlp max_length 100000'.format(row['id'],len(content_words)))
                delete_ids.append(index)
                continue
        # print('ignore {} sentence, due to due to title length is 1, truncated {} sentence, due to doc length exceed {}'
        #       .format(ignore_count, truncated_count, config.truncated_source_length))
        if config.makevocab:
            print("Writing vocab file...")
            with open(config.vocab_path, 'w') as writer:
                for word, count in vocab_counter.most_common(config.vocab_max_size):
                    writer.write(word + ' ' + str(count) + '\n')
            print("Finished writing vocab file")
        print('ignore sentence has %d'%(len(delete_ids)))
        df.drop(delete_ids,inplace=True)
        df[:-2000].to_csv(config.train_path,sep='\t', header=None,index=None,encoding='utf-8')
        df[-2000:-1000].to_csv(config.val_path,sep='\t', header=None,index=None,encoding='utf-8')
        df[-1000:].to_csv(config.test_path,sep='\t', header=None,index=None,encoding='utf-8')
    nlp_tokenizer.close()

def pre_sentence_token(df, lower=True, makevocab=True):
    if makevocab:
        vocab_counter = Counter()
    all_titles = []
    all_contents = []
    file_i = 1
    for index, row in df.iterrows():
        try:
            title = str(row['title']).strip("'<>() ")
            content = str(row['content']).strip("'<>() ")
            if lower:
                title = title.lower()
                content = content.lower()
            title_words = word_tokenizer(title)
            if len(title_words)==1:
                print('ignore id: {} sentence...,due to title length is 1'.format(row['id']))
                continue
            if makevocab:
                vocab_counter.update(title_words)
            contents = [] # 存放内容每句的句子
            content_sentences = sentence_tokenizer(content)
            for content_s in content_sentences:
                content_words = word_tokenizer(content_s)
                if makevocab:
                    vocab_counter.update(content_words)
                contents.append(content_words)
            ### bulid Vocab
            if title_words and contents:
                # 标题和内容都不为空
                all_titles.append(title_words)
                all_contents.append(contents)
            if index % 5000 == 0:
                print('has deal with %d sentence...' % index)
            # 每100000存储一个文件
            if index % 100000 == 0:
                data = {}
                data['article'] = all_contents
                data['abstract'] = all_titles
                util.save_object(data, './preprocessed/train/all_data_{}.pickle'.format(file_i))
                file_i += 1
                # 清空data,和数组
                all_contents = []
                all_titles = []
                data.clear()
        except Exception:
            print('id :{} sentence can not deal with, because the sentence has exception...'.format(row['id']))
            continue
    # save data
    data = {}
    data['article'] = all_contents
    data['abstract'] = all_titles
    util.save_object(data, './preprocessed/train/all_data_{}.pickle'.format(file_i))
    print("Writing vocab file...")
    with open('./preprocessed/all_data_vocab_200000', 'w') as writer:
        for word, count in vocab_counter.most_common(200000):
            writer.write(word + ' ' + str(count) + '\n')
    nlp_tokenizer.close()

def word_tokenizer(text):
    text_list = nlp_tokenizer.word_tokenize(text)
    res = []
    for word in text_list:
        if word != " " and word not in all_punc:
            res.append(word)
    return res

def sentence_tokenizer(text):
    return sent_tokenizer.tokenize(text)

###################### main step ######################

# step 1 train word token
# config = util.read_config('../configs/preprocess_data.yaml')
# dfs = []
# for i in range(9):
#     file_n = './data/bytecup2018/bytecup.corpus.train.%d.txt' % (i)
#     df = util.read_json_to_pd(file_n, max_lines = config.max_lines)
#     dfs.append(df)
#     print('load %d file...'%i)
# all_df = pd.concat(dfs)
# # shuffle
# all_df = all_df.sample(frac=1,random_state=config.rand_state).reset_index(drop=True)
# pre_word_token(all_df, config)

# step 2 validation_set word token
# file_n = './bytecup2018/bytecup.corpus.test_set.txt'
# test_df = util.read_json_to_pd(file_n, max_lines = 1000, is_contain_title = False)
# pre_word_token(test_df, config=None, test=True, lower=True, is_make_title=True)


######## fast_abstrcative #######
# import pandas as pd
# dfs = []
# for i in range(9):
#     file_n = './bytecup2018/bytecup.corpus.train.%d.txt' % (i)
#     df = util.read_json_to_pd(file_n, max_lines = 0)
#     dfs.append(df)
#     print('load %d file...'%i)
# all_df = pd.concat(dfs)
# print('去重前: ',all_df.shape)
# all_df = all_df.sample(frac=1,random_state=123).reset_index(drop=True)
# # 去重
# all_df.drop_duplicates(['title', 'content'], inplace=True)
# all_df.reset_index(drop=True)
# print('去重后: ',all_df.shape)
# pre_sentence_token(all_df, lower=True, makevocab=True)

# data = {}
# data['article'] = []
# data['abstract'] = []
# for i in range(1,9):
#     dic = util.load_object('./preprocessed/train/all_data_{}.pickle'.format(i))
#     print(i, len(dic['article']), len(dic['abstract']))
#     data['article'] += dic['article']
#     data['abstract'] += dic['abstract']
# print(len(data['article']))
# print(len(data['abstract']))
#
# ## 存储val
# val_dict = {}
# val_dict['article'] = data['article'][-1000:]
# val_dict['abstract'] = data['abstract'][-1000:]
# util.save_object(val_dict, './preprocessed/train/val_data.pickle')
# val_dict.clear()
#
# for i in range(7):
#     start_i = i * 100000
#     end_i = (i+1) * 100000
#     t_d = {}
#     if i==6:
#         t_d['article'] = data['article'][:-1000][start_i:]
#         t_d['abstract'] = data['abstract'][:-1000][start_i:]
#     else:
#         t_d['article'] = data['article'][:-1000][start_i: end_i]
#         t_d['abstract'] = data['abstract'][:-1000][start_i:end_i]
#     util.save_object(t_d, './preprocessed/train/train_data_{}.pickle'.format(i+1))
#     print('finish {}'.format(i))
#     t_d.clear()












