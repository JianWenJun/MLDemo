#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/6 上午10:41 
# @Author : ComeOnJian 
# @File : generate_submit.py.py 

import jieba.posseg as pseg
import pandas as pd
import numpy as np
import jieba
import re
import util
import train_model

# file path
stop_words_path = './data/stopword.txt'

# data path
all_data_path = './data/all_docs.txt'
train_data_path = './data/train_1000.csv'
test_data_path = './data/test_107295.csv'
train_candidate_path = './data/train_1000_candidate_add_title.pickle'
test_candidate_path = './data/test_107295_candidate_add_title.pickle'
# train_candidate_path = './data/train_1000_candidate.pickle'
# test_candidate_path = './data/test_107295_candidate.pickle'

# custom dict path
renming_dict_path = './data/mingxing.dict'
zhongyi_dict_path = './data/zyi.dict'
yinshi_dict_path = './data/yinshi.dict'
yaoping_dict_path = './data/yaoping.dict'

def extract_title_doc(id,title, stop_words, words_prob):
    # 书名号规则
    title_tag = re.findall(r"《(.+?)》", title)
    title_tag = [i.replace(",", "") for i in title_tag]
    for tag in title_tag:
        title = title.replace('《%s》'%(tag), 'TAG')
    words_title = list(pseg.cut(title, HMM=True))
    title_key_words = []
    title_key_not_words = []
    for word, pos in words_title:
        if len(word) > 1 and word not in stop_words:
            # 名字
            if pos == 'nr':
                if word not in title_tag:
                    title_tag.append(word)
                if word in words_prob:
                    words_prob.pop(word)
                if len(title_tag) == 2:
                    return title_tag
            # 其他名词
            elif pos in ['n','nz','ns','nt','j','eng','m']:
                if pos == 'eng' and word == 'TAG':
                    continue
                # if pos == 'eng' and len(word)>= 15:
                #     continue
                if word in words_prob:
                    if (word, words_prob[word]) not in title_key_words:
                        title_key_words.append((word, words_prob[word]))
                else:
                    title_key_not_words.append(word)
    title_key_words_sorted = sorted(title_key_words,key=lambda x: x[1],reverse=True)

    if len(title_tag) == 0:
        # title_key_words_sorted中抽两个,不够从doc_key_words_sorted中补充
        res = get_key_from_title(id,2,title_key_words_sorted,words_prob)
        if len(res) == 0:
            return title_key_not_words[:2]
        else:
            title_tag += res
        return title_tag

    if len(title_tag) == 1:
        res = get_key_from_title(id, 1, title_key_words_sorted, words_prob)
        if len(res) == 0:
            return title_tag + title_key_not_words[:1]
        else:
            title_tag += res
        return title_tag

    if len(title_tag) >= 2:
        return title_tag[:2]

def get_key_from_title(id, num, title_key_words_sorted, words_prob):
    """
    :param num: 需要抽取的关键词个数，且需要满足prob>0.5
    :param title_key_words_sorted:
    :param doc_key_words_sorted:
    """
    res_title = []
    if len(title_key_words_sorted) == 0:
        # 从 doc_key_words_sorted 中提取两个两个
        doc_key_words_sorted = sorted(words_prob.items(), key=lambda x: x[1], reverse=True)
        res = get_key_from_doc(num, doc_key_words_sorted)
        res_title += res
        return res_title

    if len(title_key_words_sorted) == 1:
        prob = title_key_words_sorted[0][1]
        if prob > 0.5:
            res_title.append(title_key_words_sorted[0][0])
            num -= 1
            if title_key_words_sorted[0][0] not in words_prob:
                print('index %s --- value %s' % (id,title_key_words_sorted[0][0]))
            else:
                words_prob.pop(title_key_words_sorted[0][0])
            # words_prob.pop(title_key_words_sorted[0][0])

        doc_key_words_sorted = sorted(words_prob.items(), key=lambda x: x[1], reverse=True)
        res = get_key_from_doc(num,doc_key_words_sorted)
        res_title += res
        return res_title

    if len(title_key_words_sorted) >= 2:
        for index,(word, prob)in enumerate(title_key_words_sorted):
            if prob > 0.5:
                res_title.append(title_key_words_sorted[index][0])
                if title_key_words_sorted[index][0] not in words_prob:
                    print('index %s --- value %s'% (id,title_key_words_sorted[index][0]))
                else:
                    words_prob.pop(title_key_words_sorted[index][0])
                num -= 1

            if num == 0:
                return res_title
        if num > 0:
            doc_key_words_sorted = sorted(words_prob.items(), key=lambda x: x[1], reverse=True)
            res = get_key_from_doc(num,doc_key_words_sorted)
            res_title += res
            return res_title

def get_key_from_doc(num, doc_key_words_sorted):
    """
    从doc 中需要抽取num个关键词
    :param num:
    :param doc_key_words_sorted:
    """
    res = []
    if num == 0:
        return res
    if len(doc_key_words_sorted) == 0:
        return res  # 0
    if len(doc_key_words_sorted) == 1:
        res.append(doc_key_words_sorted[0][0])
        return res
    if len(doc_key_words_sorted) >= 2:
        res.append(doc_key_words_sorted[0][0])
        if num == 1:
            return res
        if num == 2:
            res.append(doc_key_words_sorted[1][0])
        return res

def main():

    # step 1 模型
    train_data = pd.read_csv(train_data_path, sep='\001', header=None)
    train_data.columns = ['id', 'title', 'doc', 'key_words']
    train_candidates = util.load_object(train_candidate_path)
    Featutes, labels = train_model.build_train_sample(train_data, train_candidates)
    print(np.sum(labels))
    print(Featutes.shape)
    dt = train_model.train_class_model(Featutes, labels)

    # test
    test_data = pd.read_csv(test_data_path, sep='\001', header=None)
    stop_words = util.stopwordslist(stop_words_path)
    test_data.columns = ['id', 'title', 'doc']
    ids = test_data['id'].tolist()
    titles = test_data['title'].tolist()
    docs = test_data['doc'].tolist()
    test_candidates = util.load_object(test_candidate_path)
    sample_label_probs = train_model.get_test_sample_prob(dt, test_data, test_candidates)

    # util.save_object(sample_label_probs, './data/sample_labels_probs_add_title.pickle')
    # sample_label_probs = util.load_object('./data/sample_labels_probs_add_title.pickle')
    # sample_label_probs = util.load_object('./data/sample_title_doc_labels_probs1.pickle')
    with open('last_summit2.csv','w') as file:
        file.write('id,label1,label2\n')
        for (id, title, doc, words_prob) in zip(ids, titles, docs, sample_label_probs):
            if id == 'D087215':
                print('test......')
            if id == 'D087268':
                print('test......')

            title = str(title).strip()
            last_labes = extract_title_doc(id, title, stop_words, words_prob)
            labels_str = ",".join(last_labes)
            if len(last_labes) <= 1:
                labels_str += ','
            file.write(id + "," + labels_str)
            file.write("\n")
if __name__ == '__main__':
    # load custom dict
    jieba.load_userdict(renming_dict_path)
    jieba.load_userdict(zhongyi_dict_path)
    jieba.load_userdict(yinshi_dict_path)
    jieba.load_userdict(yaoping_dict_path)
    main()



