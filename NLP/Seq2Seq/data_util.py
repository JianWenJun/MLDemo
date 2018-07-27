#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/7/20 上午9:02 
# @Author : ComeOnJian 
# @File : data_util.py
# 参考https://kingsleyhsu.github.io/2017/10/26/20171026-NLP%E9%A2%84%E5%A4%84%E7%90%86/
import re
import jieba
import numpy as np
from tensorflow.python.platform import gfile
import os
from zhon.hanzi import punctuation

#################### 清洗Sougou新闻语料库的工作 ####################
# 文本中特殊的标志
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")
root_path = os.path.abspath(os.path.join(os.getcwd(),'../data/text_summar'))

##############################  全角和半角文本转换 ##############################################
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring
    return text
##############################  文本清理工作 ##############################################
def remove_url(text):
	r=u'((https?|ftp|file)://){,1}[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
	return re.sub(r,'TAG_URL', text)

def remove_pun_ch(text):
	return re.sub(ur"[%s]"%punctuation , "", text.decode('utf-8'))

def remove_pun_en(text):
	r=u'[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
	return re.sub(r,'', text)

def remove_date(text):
	r=u'\d{1,4}年\d{1,2}月\d{1,2}日 |\d{1,4}年\d{1,2}月|\d{1,2}月\d{1,2}日|\d{1,4}年|\d{1,2}月|\d{1,2}日'
	return re.sub(r, 'TAG_DATE', strQ2B(text))

def remove_num(text):
	r=u'\d{1,}'
	return re.sub(r,'TAG_NUMBER', text)

def remove_num_en(text):
	r=u'([\uff01-\uff5e]){1,}'
	return re.sub(r,'TAG_NUM_EN', text)

def remove_tag(text):
    text = remove_url(text)
    text = remove_pun_en(text)
    text = remove_pun_ch(text) #之后是Unicode
    text = remove_date(text)
    text = remove_num_en(text)
    text = remove_num(text)
    return text
##############################  文本转换为训练集和测试集的预处理工作 ##############################################
def get_title_content(content_fp,title_fp):
    """
    :param content_fp:新闻文本的内容文件路径
    :param title_fp:新闻文本的标题文件路径
    :return:list格式的content，title
    """
    data_content = []
    data_title = []
    tmp_content = []
    tmp_title = []
    with open(content_fp,'r') as be_content_f:
        for line in be_content_f.readlines():
            #截取content的内容，去除<content></content>标签
            tmp_content.append(line[9:-11])
    with open(title_fp, 'r') as be_title_f:
        for line in be_title_f.readlines():
            tmp_title.append(line[14:-16])

    indices = len(tmp_content)
    a = np.arange(indices)
    b = np.random.permutation(a)

    for i in b:
        if(tmp_content[i] and tmp_title[i]):
            data_content.append(tmp_content[i])
            data_title.append(tmp_title[i])
    return data_content,data_title

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def jieba_tokenizer(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    return sentence_seged

def create_vocab(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """
    构建词典
    :param vocabulary_path:词典路径
    :param data_path:训练集文本路径
    :param max_vocabulary_size:词典max size
    :param tokenizer:词条化函数，若=None，使用basic_tokenizer
    :param normalize_digits:if true, 所有数字用0代替
    :return:
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """
    初始化词典
	假设词典文件如下：
		dog
		cat
    :param vocabulary_path:词典所在路径
    :return:
    vocabulary ={"dog": 0, "cat": 1}, reversed-vocabulary=["dog", "cat"].
    vocabulary ：词典类型
	reversed vocabulary ：list
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
	'''
	将文件转换成id
	句子"I have a dog" 用词典{"I": 1, "have": 2,"a": 4, "dog": 7"}返回[1, 2, 4, 7]
	输入:
		sentence: 句子用bytes格式
		vocabulary: 词典
		tokenizer:词条化函数
		normalize_digits: 是否数字化
	返回:
		id号
	'''

	if tokenizer:
		words = tokenizer(sentence)
	else:
		words = basic_tokenizer(sentence)
	if not normalize_digits:
		return [vocabulary.get(w, UNK_ID) for w in words]
	# Normalize digits by 0 before looking words up in the vocabulary.
	return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
	'''
	将data文件转换成idsokenize
	输入：
		data_path: data 文件路径
		target_path:ids文件路径
		vocabulary_path: 词汇表路径
		tokenizer:词条化函数
		normalize_digits: 数字是否处理
	'''
	vocab, _ = initialize_vocabulary(vocabulary_path) #vocab=dict{"dog": 0, "cat": 1}
	data_f = open(data_path,"r")
	tokens_f = open(target_path,"w+")
	counter = 0
	for line in data_f.readlines():
		counter += 1
		if counter % 100000 == 0:
			print("  tokenizing line %d" % counter)
		token_ids = sentence_to_token_ids(line, vocab, tokenizer,normalize_digits)
		tokens_f.write(" ".join([str(tok) for tok in token_ids]) + "\n")#写到文件


def get_train_dev_sets(data_content, data_title, train_rate, dev_rate,
                       tr_con_path, tr_title_path,
                       dev_con_path, dev_title_path,
                       test_con_path, test_title_path):
    """
    按照train_rate，dev_rate切分train_sets，dev_sets和test_sets
    :param data_content:
    :param data_title:
    :param train_rate:
    :param dev_rate:
    :param tr_con_path:
    :param tr_title_path:
    :param dev_con_path:
    :param dev_title_path:
    :param test_con_path:
    :param test_title_path:
    :return:
    """
    tr_con_f = open(tr_con_path, "w+")
    tr_title_f = open(tr_title_path, "w+")
    dev_con_f = open(dev_con_path, "w+")
    dev_title_f = open(dev_title_path, "w+")
    test_con_f = open(test_con_path, "w+")
    test_title_f = open(test_title_path, "w+")

    line_num = len(data_content)
    train_num = int(line_num * train_rate)
    dev_num = int(line_num * dev_rate)

    for i in range(0, train_num):
        tr_con_f.write(data_content[i])
        tr_title_f.write(data_title[i])

    for i in range(train_num, dev_num + train_num):
        dev_con_f.write(data_content[i])
        dev_title_f.write(data_title[i])

    for i in range(dev_num + train_num, line_num):
        test_con_f.write(data_content[i])
        test_title_f.write(data_title[i])
    tr_con_f.close()
    tr_title_f.close()
    dev_con_f.close()
    dev_title_f.close()
    test_con_f.close()
    test_title_f.close()
    return (tr_con_path, tr_title_path, dev_con_path, dev_title_path, test_con_path, test_title_path)

def prepare_headline_data(data_dir, vocabulary_size, tokenizer=None):
    """
    为模型训练准备数据
    :param data_dir:数据存储的目录
    :param vocabulary_size:词典的大小
    :param tokenizer：词条化函数
    :return:
    """
    train_path = os.path.join(root_path, "train")
    src_train_path = os.path.join(train_path, "content-train-origin.txt")
    dest_train_path = os.path.join(train_path, "title-train-origin.txt")

    dev_path = os.path.join(root_path, "dev")
    src_dev_path = os.path.join(dev_path, "content-dev-origin.txt")
    dest_dev_path = os.path.join(dev_path, "title-dev-origin.txt")

    # 创建vocab
    vocab_path = data_dir + '/vocab'
    create_vocab(vocab_path,src_train_path,vocabulary_size,tokenizer)

    # 创建 token ids for the training data.
    src_train_ids_path = os.path.join(train_path, "content_train_id")
    dest_train_ids_path = os.path.join(train_path, "title_train_id")
    data_to_token_ids(src_train_path, src_train_ids_path, vocab_path, tokenizer)
    data_to_token_ids(dest_train_path, dest_train_ids_path, vocab_path, tokenizer)

    # 创建 token ids for the development data.
    src_dev_ids_path = os.path.join(dev_path, "content_dev_id")
    dest_dev_ids_path = os.path.join(dev_path, "title_dev_id")
    data_to_token_ids(src_dev_path, src_dev_ids_path, vocab_path, tokenizer)
    data_to_token_ids(dest_dev_path, dest_dev_ids_path, vocab_path, tokenizer)

if __name__ == '__main__':
    content_fp = root_path + '/corpus_50.txt'
    title_fp = root_path + '/corpus_title_50.txt'
    jieba.load_userdict(root_path+'/dict.txt')
    print(content_fp)

    train_path = os.path.join(root_path, "train")
    src_train_path = os.path.join(train_path, "content-train-origin.txt")
    dest_train_path = os.path.join(train_path, "title-train-origin.txt")

    dev_path = os.path.join(root_path, "dev")
    src_dev_path = os.path.join(dev_path, "content-dev-origin.txt")
    dest_dev_path = os.path.join(dev_path, "title-dev-origin.txt")

    test_path = os.path.join(root_path, "test")
    src_test_path = os.path.join(test_path, "content-test-origin.txt")
    dest_test_path = os.path.join(test_path, "title-test-origin.txt")

    # step1 获取出文本内容
    data_content,data_title = get_title_content(content_fp,title_fp)
    indexs = np.arange(len(data_content))

    # step2 去除tag
    for index,content,title in zip(indexs,data_content,data_title):
        data_content[index] = remove_tag(content).encode('utf-8')
        data_title[index] = remove_tag(title).encode('utf-8')

    # step3 划分数据，训练集，验证集，测试集
    get_train_dev_sets(data_content,data_title,train_rate=0.7,dev_rate=0.1,
                       tr_con_path=src_train_path,tr_title_path=dest_train_path,
                       dev_con_path=src_dev_path,dev_title_path=dest_dev_path,
                       test_con_path=src_test_path,test_title_path=dest_test_path
                       )

    # step4
    prepare_headline_data(root_path,vocabulary_size=80000,tokenizer=jieba_tokenizer)
    pass
