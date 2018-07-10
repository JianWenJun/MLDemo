#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/6/15 下午10:50 
# @Author : ComeOnJian 
# @File : data_prepare.py 


import jieba
from util import *
import json
import re
from collections import defaultdict
from gensim.models import word2vec
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# file paths
train_data_all = 'atec_nlp_sim_train_all.csv'
train_all = 'atec_nlp_sim_train_0.6.csv'
test_all = 'atec_nlp_sim_test_0.4.csv' #6550个为1的label
stop_words_path = 'stop_words.txt'
dict_path = 'dict_all.txt'
spelling_corrections_path = 'spelling_corrections.json'
w2v_model_path = 'train_corpus.model'
w2v_vocab_path = 'train_corpus_vocab.txt'

# data


#param
embedding_size = 300
max_sentence_length = 20
max_word_length = 25
# os.path.join(project.aux_dir,'fasttext','')
max_vovab_size = 100000

#################### 文本的清理工作 ####################
def preprocessing(data_df,fname):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:
    """
    # 加载停用词
    stopwords = load_stopwordslist(project.aux_dir + stop_words_path)
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(project.aux_dir + spelling_corrections_path)

    re_object = re.compile(r'\*+') #去除句子中的脱敏数字***，替换成一
    vocabs = defaultdict(int)# 记录词汇表词频
    for index, row in data_df.iterrows():
        # 每1000个打印一下句子的词向量
        if index != 0 and index % 2000 == 0:
            print("{:,}  {}-sentence embedding.".format(index,fname))
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["s1", "s2"]:
            # 替换掉脱敏的数字
            re_str = re_object.subn(u"十一",unicode(row[col_name],'utf-8'))
            # 纠正一些词
            spell_corr_str = transform_other_word(re_str[0],spelling_corrections)
            # 分词
            seg_str = seg_sentence(spell_corr_str, stopwords)
            for word in seg_str.split(" "):
                vocabs[word] = vocabs[word] + 1
                data_df.at[index, col_name] = seg_str

    data_df.to_csv(project.preprocessed_data_dir + '{}.csv'.format(fname), sep='\t', header=None,index=None,encoding='utf-8')
    project.save(project.preprocessed_data_dir + '{}.pickle'.format(fname),vocabs)
    del data_df

def seg_sentence(sentence,stop_words):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    sentence_seged = jieba.cut(sentence.strip())
    out_str = ""
    for word in sentence_seged:
        if word not in stop_words:
            if word != " ":
                out_str += word
                out_str += " "
    return out_str

def load_stopwordslist(filepath):
    """
    加载停用词
    :param filepath:停用词文件路径
    :return:
    """
    with io.open(filepath,"r",encoding="utf-8") as file:
        stop_words = [line.strip() for line in file]
        return stop_words

def load_spelling_corrections(filepath):
    with io.open(filepath,"r",encoding="utf-8") as file:
        spelling_corrections = json.load(file)
        return spelling_corrections

def load_doubt_words(filpath):
    """
    加载疑问词
    :param filpath:
    :return:
    """
    with io.open(filpath,'r',encoding="utf-8") as file:
        doubt_words = [line.strip() for line in file]
        return doubt_words

def transform_other_word(str_text,reg_dict):
    for token_str,replac_str in reg_dict.items():
        str_text = str_text.replace(token_str, replac_str)
    return str_text

def strip_why(rawq):
    rawq = re.sub('为什么|为何|为啥|为么|为撒|咋个|为什|怎么回事|是什么原因|什么原因', '', rawq)
    if re.match(r'怎么.*(不|没|了|只|会|又|要|老|总|才|是)',rawq):
        rawq = re.sub('怎么', '', rawq)
    return rawq

def strip_how(rawq):
    rawq = re.sub('怎么办|咋办', '', rawq)
    return rawq

#################### 文本的Embeding工作 ####################
def process_save_embedding_wv(nfile,type = 1,isStore_ids = False):
    """
    :param type: 词向量的选择：1，知乎，2，训练集 3 知乎+训练集
    :return:
    """
    w2v_path = project.aux_dir + 'sgns.zhihu.bigram'

    if type == 2:
        w2v_path = project.aux_dir + 'train_all_data.bigram'

    # vocabs_path = project.preprocessed_data_dir + 'data_all_seg.pickle'

    tokenizer = Tokenizer(
        num_words=max_vovab_size,
        split=' ',
        lower=False,
        char_level=False,
        filters=''
    )
    # 加载所有的词汇表训练集和测试集
    pre_deal_train_df = pd.read_csv(project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                    names=["index", "s1", "s2", "label"],
                                    header=None,encoding='utf-8',
                                    sep='\t')
    pre_deal_test_df = pd.read_csv(project.preprocessed_data_dir + 'test_0.4_seg.csv',
                                   names=["index", "s1", "s2", "label"],
                                   header=None,encoding='utf-8',
                                   sep='\t',
                                   )
    texts = []
    texts_s1_test = pre_deal_test_df['s1'].tolist()
    texts_s2_test = pre_deal_test_df['s2'].tolist()

    texts_s1_train = pre_deal_train_df['s1'].tolist()
    texts_s2_train = pre_deal_train_df['s2'].tolist()

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)

    # print pre_deal_train_df.isnull().any()
    # print pre_deal_test_df.isnull().any()
    # 生成token词典
    # tests = [u'中 国', u'矿业 大学', u'不错哦']
    tokenizer.fit_on_texts(texts)

    # 生成各个词对应的index列表
    s1_train_ids = tokenizer.texts_to_sequences(texts_s1_train)
    s2_train_ids = tokenizer.texts_to_sequences(texts_s2_train)


    s1_test_ids = tokenizer.texts_to_sequences(texts_s1_test)
    s2_test_ids = tokenizer.texts_to_sequences(texts_s2_test)

    num_words_dict = tokenizer.word_index

    # 训练集的词汇表的词向量矩阵,行数为最大值+1,形式为：index->vec
    embedding_matrix = 1 * np.random.randn(len(num_words_dict) + 1, embedding_size)
    embedding_matrix[0] = np.random.randn(embedding_size)

    # 加载预训练的词向量w2v
    print 'load w2v_model...'
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    print 'finish w2v_model...'

    if type == 3:
        w2v_path2 = project.aux_dir + 'train_all_data.bigram'
        w2v_model2 = KeyedVectors.load_word2vec_format(w2v_path2, binary=False)
    count = 0
    for word,index in num_words_dict.items():
        if word in w2v_model.vocab:
            embedding_matrix[index] = w2v_model.word_vec(word)
            count = count +1
        else:
            if type == 3:
                if word in w2v_model2.vocab:
                    embedding_matrix[index] = w2v_model2.word_vec(word)
                    count = count + 1
    print('total {}, word in model have {}'.format(len(num_words_dict),count))

    project.save(project.aux_dir + nfile,embedding_matrix)

    if isStore_ids:
        s1_train_ids_pad = sequence.pad_sequences(s1_train_ids,maxlen=max_sentence_length)
        s2_train_ids_pad = sequence.pad_sequences(s2_train_ids,maxlen=max_sentence_length)

        s1_test_ids_pad = sequence.pad_sequences(s1_test_ids,maxlen=max_sentence_length)
        s2_test_ids_pad = sequence.pad_sequences(s2_test_ids,maxlen=max_sentence_length)

        project.save(project.preprocessed_data_dir + 's1_train_ids_pad.pickle',s1_train_ids_pad)
        project.save(project.preprocessed_data_dir + 's2_train_ids_pad.pickle',s2_train_ids_pad)
        project.save(project.preprocessed_data_dir + 's1_test_ids_pad.pickle',s1_test_ids_pad)
        project.save(project.preprocessed_data_dir + 's2_test_ids_pad.pickle',s2_test_ids_pad)
    print('finish')

def process_save_char_embedding_wv(isStore_ids = False):
    data_local_df = pd.read_csv(project.data_dir + train_all, sep='\t', header=None,names=["index", "s1", "s2", "label"])
    data_test_df = pd.read_csv(project.data_dir + test_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
    w2v_char_path = project.aux_dir + 'train_char_all__data.bigram'
    w2v_char_model = KeyedVectors.load_word2vec_format(w2v_char_path, binary=False)
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(project.aux_dir + spelling_corrections_path)
    re_object = re.compile(r'\*+')  # 去除句子中的脱敏数字***，替换成一
    char_vocabs = project.load(project.preprocessed_data_dir + 'train_all_char_vocabs.pickle')
    data_df_list = [data_local_df,data_test_df]

    embedding_word_matrix = 1 * np.random.randn((len(char_vocabs) + 1), embedding_size)
    embedding_word_matrix[0] = np.random.randn(embedding_size)

    for word,index in char_vocabs.items():
        if word in w2v_char_model.vocab:
            embedding_word_matrix[index] = w2v_char_model.word_vec(word)
            if index % 100 == 0:
                print 'char {}'.format(index)

    project.save(project.aux_dir + 'train_all_char_embedding_matrix.pickle',embedding_word_matrix)

    for data_df in data_df_list:
        for index, row in data_df.iterrows():
            # 每1000个打印一下句子的词向量
            if index != 0 and index % 5000 == 0:
                print("{:,}sentence word embedding.".format(index))
            # 分别遍历每行的两个句子，并进行分词处理
            for col_name in ["s1", "s2"]:
                # 替换掉脱敏的数字
                re_str = re_object.subn(u"十一",unicode(row[col_name],'utf-8'))
                # 纠正一些词
                spell_corr_str = transform_other_word(re_str[0],spelling_corrections)
                spell_corr_str = list(spell_corr_str)
                indexs = []
                for char in spell_corr_str:
                    if char in char_vocabs:
                        indexs.append(char_vocabs[char])
                    else:
                        if not char.strip()==u"":
                            indexs.append(0)
                data_df.at[index, col_name] = indexs

    if isStore_ids:
        s1_train_ids_pad = sequence.pad_sequences(data_local_df['s1'],maxlen=max_word_length)
        s2_train_ids_pad = sequence.pad_sequences(data_local_df['s2'],maxlen=max_word_length)

        s1_test_ids_pad = sequence.pad_sequences(data_test_df['s1'],maxlen=max_word_length)
        s2_test_ids_pad = sequence.pad_sequences(data_test_df['s2'],maxlen=max_word_length)

        project.save(project.preprocessed_data_dir + 's1_train_char_ids_pad.pickle',s1_train_ids_pad)
        project.save(project.preprocessed_data_dir + 's2_train_char_ids_pad.pickle',s2_train_ids_pad)
        project.save(project.preprocessed_data_dir + 's1_test_char_ids_pad.pickle',s1_test_ids_pad)
        project.save(project.preprocessed_data_dir + 's2_test_char_ids_pad.pickle',s2_test_ids_pad)
    print('finish')

#################### 使用训练集train_all做 pre_train word embedding ####################
def pre_train_w2v(binary = False):
    """
    利用已经训练集训练词向量
    :param nfile_corpus:已经分好词的文本路径，如"train_segment.corpus"
    :param binary:将词向量表是否存储为二进制文件
    :return:
    """
    # 加载所有的词汇表训练集和测试集
    pre_deal_train_df = pd.read_csv(project.preprocessed_data_dir + 'train_0.6_seg.csv',
                                    names=["index", "s1", "s2", "label"],
                                    header=None, encoding='utf-8',
                                    sep='\t')
    pre_deal_test_df = pd.read_csv(project.preprocessed_data_dir + 'test_0.4_seg.csv',
                                   names=["index", "s1", "s2", "label"],
                                   header=None, encoding='utf-8',
                                   sep='\t',
                                   )
    texts = []
    texts_s1_test = [line.strip().split(" ") for line in pre_deal_test_df['s1'].tolist()]
    texts_s2_test = [line.strip().split(" ") for line in pre_deal_test_df['s2'].tolist()]

    texts_s1_train = [line.strip().split(" ") for line in pre_deal_train_df['s1'].tolist()]
    texts_s2_train = [line.strip().split(" ") for line in pre_deal_train_df['s2'].tolist()]

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)

    model = word2vec.Word2Vec(sentences=texts,size=300,window=2,min_count=3,workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + "train_all_data.bigram",binary=binary,fvocab=None)

def pre_train_char_w2v(binary = False):
    data_local_df = pd.read_csv(project.data_dir + train_all, sep='\t', header=None,names=["index", "s1", "s2", "label"])
    data_test_df = pd.read_csv(project.data_dir + test_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
    # 加载停用词
    stopwords = load_stopwordslist(project.aux_dir + stop_words_path)
    # 加载拼写错误替换词
    spelling_corrections = load_spelling_corrections(project.aux_dir + spelling_corrections_path)

    re_object = re.compile(r'\*+')  # 去除句子中的脱敏数字***，替换成一
    data_df_list = [data_local_df,data_test_df]
    texts = []
    char_vocabs = {}
    char_index = 1
    for data_df in data_df_list:
        for index, row in data_df.iterrows():
            if index != 0 and index % 5000 == 0:
                print("{:,} sentence word embedding.".format(index))
            # 分别遍历每行的两个句子，并进行分词处理
            for col_name in ["s1", "s2"]:
                # 替换掉脱敏的数字
                re_str = re_object.subn(u"十一", unicode(row[col_name], 'utf-8'))
                # 纠正一些词
                spell_corr_str = transform_other_word(re_str[0], spelling_corrections)
                spell_corr_str = list(spell_corr_str)
                for char in spell_corr_str:
                    if char not in char_vocabs and char not in stopwords and not char.strip()==u"":
                        char_vocabs[char] = char_index
                        char_index = char_index + 1
                texts.extend(spell_corr_str)

    model = word2vec.Word2Vec(sentences=texts,size=300,window=3,workers=2)
    model.wv.save_word2vec_format(fname=project.aux_dir + "train_char_all__data.bigram",binary=binary,fvocab=None)
    project.save(project.preprocessed_data_dir + 'train_all_char_vocabs.pickle', char_vocabs)

if __name__ == '__main__':

    # step 1 # 预处理文本
    jieba.load_userdict(project.aux_dir + dict_path)
    data_local_df = pd.read_csv(project.data_dir + train_all, sep='\t', header=None,names=["index", "s1", "s2", "label"])

    data_test_df = pd.read_csv(project.data_dir + test_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])
    data_all_df = pd.read_csv(project.data_dir + train_data_all, sep='\t', header=None, names=["index", "s1", "s2", "label"])

    pre_train_char_w2v()
    #
    preprocessing(data_local_df,'train_0.6_seg')
    preprocessing(data_test_df,'test_0.4_seg')
    preprocessing(data_all_df,'data_all_seg')

    # 保存label
    project.save(project.features_dir + 'y_0.4_test.pickle', data_test_df['label'].tolist())
    project.save(project.features_dir + 'y_0.6_train.pickle', data_local_df['label'].tolist())
    project.save(project.features_dir + 'y_train.pickle', data_all_df['label'].tolist())

    # step 2
    pre_train_w2v()


    # step 3
    process_save_embedding_wv('train_all_w2v_embedding_matrix.pickle',type=2,isStore_ids=True)
    # process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=2,isStore_ids=False)
    # process_save_embedding_wv('zhihu_w2v_embedding_matrix.pickle',type=3,isStore_ids=False)

    # step 4 char wordembedding
    process_save_char_embedding_wv(isStore_ids=True)



