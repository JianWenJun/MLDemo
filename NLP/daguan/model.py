#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/8/29 上午11:06 
# @Author : ComeOnJian 
# @File : model.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder_cnn(nn.Module):
    """
    对源句子进行编码操作
    """
    def __init__(self,config,filter_sizes,filter_nums,vocab_size,embedding=None):
        super(encoder_cnn,self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size,config.emb_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=filter_num,
                      kernel_size=(fliter_size, config.emb_size),
                      padding=((fliter_size//2, 0)))
             for (filter_num,fliter_size) in zip(filter_nums,filter_sizes)])

        # 卷积层后添加BN
        self.bns = None
        if config.batchNormal:
            self.bns = nn.ModuleList([nn.BatchNorm2d(filter_num) for filter_num in filter_nums])
        #全连接层
        if config.highway:
            sum_filters = sum(filter_nums)
            self.h_layer = nn.Linear(sum_filters,sum_filters)
            self.transform_gate_layer = nn.Linear(sum_filters,sum_filters)
        self.config = config

    def forward(self, inputs):
        embs = self.embedding(inputs) # [batch, seq_length] -> [batch,seq_length,emb_size]
        x = torch.unsqueeze(embs,1) # [batch,seq_length,emb_size] -> [batch,1,seq_length,emb_size] add input channel

        xs = []
        if self.bns is not None:
            for (conv,bn)in zip(self.convs,self.bns):
                x2 = F.relu(conv(x))  # [batch,1,seq_length,emb_size] -> [batch,filter_num,seq_length,1]
                x2 = bn(x2) # [batch,filter_num,seq_length,1] -> [batch,filter_num,seq_length,1]
                x2 = torch.squeeze(x2,-1)  # [batch,filter_num,seq_length,1] -> [batch,filter_num,seq_length]
                x2 = F.max_pool1d(x2,x2.size(2)).squeeze(2) # [batch,filter_num,seq_length] -> [batch,filter_num,1] -> [batch,filter_num]
                xs.append(x2)
        else:
            for conv in self.convs:
                x2 = F.relu(conv(x))  # [batch,1,seq_length,emb_size] -> [batch,filter_num,seq_length,1]
                x2 = torch.squeeze(x2,-1)  # [batch,filter_num,seq_length,1] -> [batch,filter_num,seq_length]
                x2 = F.max_pool1d(x2,x2.size(2)).squeeze(2) # [batch,filter_num,seq_length] -> [batch,filter_num,1] -> [batch,filter_num]
                xs.append(x2)
        # xs # [batch,filter_num] * len(filter_nums) #difference filter
        pool_flat_out = torch.cat(xs,1)  # [batch,filter_num] * len(filter_nums) -> [batch,filter_num1+filter_num2+...] # 把各个filter_num相加

        # highway layout formula for https://www.cnblogs.com/bamtercelboo/p/7611380.html
        if self.config.highway:
            h = self.h_layer(pool_flat_out)
            transform_gate = F.sigmoid(self.transform_gate_layer(pool_flat_out))
            carry_gate = 1. - transform_gate  #C
            gate_transform_input = torch.mul(h,transform_gate)
            gate_carry_input = torch.mul(carry_gate,pool_flat_out)
            pool_flat_out = torch.add(gate_carry_input,gate_transform_input) #[batch,sum(filter_nums)]

        return pool_flat_out

class Text_WCCNN(nn.Module):

    def __init__(self,config, word_filter_sizes, word_filter_nums, word_vocab_size, char_filter_sizes, char_filter_nums, char_vocab_size, word_embedding=None, art_embedding=None):
        super(Text_WCCNN, self).__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(size_average=True)
        if word_embedding is not None:
            self.word_embedding = word_embedding
        else:
            self.word_embedding = nn.Embedding(word_vocab_size,config.word_emb_size)

        # article char
        self.article_encoder_cnn = encoder_cnn(self.config, char_filter_sizes, char_filter_nums, char_vocab_size, art_embedding)

        # multi cnn layer
        self.word_encoder_cnn = [nn.Sequential(
            # first layer cnn
            # [batch,1,seq_length,emb_size] -> [batch,filter_num,seq_length,1]
            nn.Conv2d(in_channels=1,out_channels=filter_num,kernel_size=(fliter_size,config.word_emb_size),padding=((fliter_size//2, 0))),
            nn.BatchNorm2d(filter_num), # [batch,filter_num,seq_length,1]
            nn.ReLU(inplace=True),

            # second layer cnn
            nn.Conv2d(in_channels=filter_num,out_channels=filter_num,kernel_size=(fliter_size,config.word_emb_size),padding=((fliter_size//2, 0))),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(config.seq_length - fliter_size * 2 + fliter_size//2+1))
        )
        for (filter_num,fliter_size) in zip(word_filter_nums,word_filter_sizes)]

        # flat layer
        self.fc = nn.Sequential(
            nn.Linear(sum(char_filter_nums) + sum(word_filter_nums),config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.linear_hidden_size, config.num_classes)
        )

    def forward(self, article, word_seg):

        word_embs = self.word_embedding(word_seg) # [batch, seq_length] -> [batch,seq_length,emb_size]
        x = torch.unsqueeze(word_embs, 1)  # [batch,seq_length,emb_size] -> [batch,1,seq_length,emb_size] add input channel

        word_convs = [word_conv(x) for word_conv in self.word_encoder_cnn]
        word_context = torch.cat(word_convs, 1) # [batch,sum(word_filter_nums)]
        article_context = self.article_encoder_cnn(article) # [batch,sum(char_filter_nums)]

        flat_out = torch.cat((word_context,article_context),1) # [batch,sum(char_filter_nums) + sum(word_filter_nums)]
        flat_out = self.fc(flat_out)
        return flat_out








