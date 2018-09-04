#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/9/1 上午12:16 
# @Author : ComeOnJian 
# @File : predict.py 
import data_analy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import model

if False:
    test = pd.read_csv('test_set.csv')
    article_dicts = data_analy.load('./data/dicts_10000_arts')
    word_dicts = data_analy.load('./data/dicts_250000_words')

    article = test['article'].values
    words = test['word_seg'].values

    test_articles_ids = data_analy.sentence_to_indexs(article,article_dicts['dict_label2id'],article_dicts['stop_word'],padding=False)
    test_words_ids = data_analy.sentence_to_indexs(words,word_dicts['dict_label2id'],word_dicts['stop_word'],padding=False)

    data_analy.save('./data/test_articles_ids.pickle',test_articles_ids)
    data_analy.save('./data/test_words_ids.pickle',test_words_ids)



# load model
config = data_analy.read_config('./config.yaml')
torch.manual_seed(config.seed)

print('loading checkpoint...\n')
check_point = torch.load('./data/best_macro_f1_checkpoint.pt')

# cuda
use_cuda = torch.cuda.is_available() and len(config.gpus) > 0
if use_cuda:
    torch.cuda.set_device(config.gpus[0])
    torch.cuda.manual_seed(config.seed)

# load model
word_filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
word_filter_nums = [100, 150, 150, 150, 150, 50, 50, 50]
char_filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
char_filter_nums = [50, 100, 100, 100, 50, 50, 50, 50]
cnn_model = getattr(model,"Text_WCCNN")(config, word_filter_sizes, word_filter_nums, config.word_vocab_size,
                         char_filter_sizes, char_filter_nums, config.char_vocab_size)

if use_cuda:
    cnn_model = cnn_model.cuda()
cnn_model.load_state_dict(check_point['model'])
# load dataset
if True:
    test_articles_ids = data_analy.load('./data/test_articles_ids.pickle')
    test_words_ids = data_analy.load('./data/test_words_ids.pickle')
    y_test = np.zeros(len(test_words_ids)).tolist()
    test_dataset = data_analy.dataset(test_articles_ids, test_words_ids, y_test)

test_dataloader = data_analy.get_loader(test_dataset,
                                         batch_size = config.batch_size,
                                         shuffle = False,
                                         num_workers = 2
                                         )

def eval(cnn_model):
    cnn_model.eval()
    y_pred = []
    for art_src, word_src, y in test_dataloader:
        if len(y) != config.batch_size:
            print('-------------------')
        art_src = Variable(art_src)
        word_src = Variable(word_src)

        if use_cuda:
            art_src = art_src.cuda()
            word_src = word_src.cuda()
        if len(config.gpus) > 1:
            output_pro = cnn_model.module.forward(art_src,word_src) # FloatTensor [batch_size,1]
        else:
            output_pro = cnn_model.forward(art_src,word_src)
        # y_pred += [item.tolist() for item in output_pro.data] # item is FloatTensor size 19
        pred_label = torch.max(output_pro, 1)[1].data.tolist() # LongTensor size 32
        y_pred += [int(item) for item in pred_label]

    return y_pred
y_pred = eval(cnn_model)
fid0=open('textcnn1.csv','w')
i=0
fid0.write("id,class"+"\n")
for item in y_pred:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()

