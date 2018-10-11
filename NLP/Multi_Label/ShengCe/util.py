#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/6 上午10:47 
# @Author : ComeOnJian 
# @File : util.py 
import re
import math
import pickle
pos_dict = {
    'n': 0,
    'nr': 1,
    'nz': 2,
    'ns': 3,
    'eng': 4,
    'nt': 5,
    'j': 6}

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def get_shuming(text):
    shuming_tag = re.findall(r"《(.+?)》", text)
    shuming_tag = [i.replace(",", "") for i in shuming_tag]
    return shuming_tag

def is_contain_char_num(text):
    """
    判断文本text的字符是否全部由数字或者字母组成
    """
    char_num = re.findall(r"[a-zA-Z0-9]+", text)
    if text in char_num:
        return True
    else:
        return False

def get_count_sentence(word,sentence):
    """
    判断word在sentence词的列表中出现的次数
    """
    first_count = 0
    for first_word, first_pos in sentence:
        if word == first_word:
            first_count += 1
    return first_count

def cal_sim(word_topic_prob, doc_topic_prob):
    # 计算两个向量的余弦相似度
    a, b, c = 0.0, 0.0, 0.0
    for t1, t2 in zip(word_topic_prob, doc_topic_prob):
        x1 = t1[1]
        x2 = t2[1]
        a += x1 * x2
        b += x1 * x1
        c += x2 * x2
    sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
    return sim

def save_object(obj,nfile):
    with open(nfile,'wb') as file:
        pickle.dump(obj,file)

def load_object(nfile):
    with open(nfile,'rb') as file:
        return pickle.load(file)

