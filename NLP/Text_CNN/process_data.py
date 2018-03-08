#!/usr/bin/python
# coding=utf-8
# @Time : 2018/3/8 下午3:02
# @Author : ComeOnJian
# @File : process_data.py

import pickle
# import word2vec
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm
data_dir = '../data/rt-polaritydata/'
google_new_vector_dir = '../data/'

##方式一，编写代码加载word2vec训练好的模型文件GoogleNews-vectors-negative300.bin
####参照word2vec.from_binary方法改写

def load_binary_vec(fname, vocab):
    word_vecs = {}
    with open(fname, 'rb') as fin:
        header = fin.readline()
        vocab_size, vector_size = list(map(int, header.split()))
        binary_len = np.dtype(np.float32).itemsize * vector_size
        # vectors = []
        for i in tqdm(range(vocab_size)):
            # read word
            word = b''
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                word += ch
            # print(str(word))
            word = word.decode(encoding='ISO-8859-1')
            if word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len), dtype=np.float32)
            else:
                fin.read(binary_len)
            # vector = np.fromstring(fin.read(binary_len), dtype=np.float32)
            # vectors.append(vector)
            # if include:
            #     vectors[i] = unitvec(vector)
            fin.read(1)  # newline
        return word_vecs

#load MR data —— Movie reviews with one sentence per review.
#Classification involves detecting positive/negative reviews
def load_data_k_cv(folder,cv=10,clear_flag=True):
    pos_file = folder[0]
    neg_file = folder[1]

    #训练集的语料词汇表,计数
    word_cab=defaultdict(float)
    revs = [] #最后的数据
    with open(pos_file,'rb') as pos_f:
        for line in pos_f:
            rev = []
            rev.append(line.decode(encoding='ISO-8859-1').strip())
            if clear_flag:
                orign_rev = clean_string(" ".join(rev))
            else:
                orign_rev = " ".join(rev).lower()
            words = set(orign_rev.split())
            for word in words:
                word_cab[word] += 1
            datum = {"y": 1,
                     "text": orign_rev,
                     "num_words": len(orign_rev.split()),
                     "spilt": np.random.randint(0, cv)}
            revs.append(datum)

    with open(neg_file,'rb') as neg_f:
        for line in neg_f:
            rev = []
            rev.append(line.decode(encoding='ISO-8859-1').strip())
            if clear_flag:
                orign_rev = clean_string(" ".join(rev))
            else:
                orign_rev = " ".join(rev).lower()
            words = set(orign_rev.split())
            for word in words:
                word_cab[word] += 1
            datum = {"y": 0,
                     "text": orign_rev,
                     "num_words": len(orign_rev.split()),
                     "spilt": np.random.randint(0, cv)}
            revs.append(datum)

    return word_cab,revs

def clean_string(string,TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

if __name__ == '__main__':
    data_folder = ['../data/rt-polaritydata/rt-polarity.pos', '../data/rt-polaritydata/rt-polarity.neg']
    w2v_file = '../data/GoogleNews-vectors-negative300.bin'
    print('load data ...')
    word_cab, revs = load_data_k_cv(data_folder)
    print('data loaded --')
    print('number of sentences: ' + str(len(revs)))
    print('size of vocab: ' + str(len(word_cab)))

    print('load word2vec vectors...')
    word_vecs = load_binary_vec(w2v_file,word_cab)
    # print (len(list(word_vecs.keys())))
    print('finish word2vec !!!')

    pass