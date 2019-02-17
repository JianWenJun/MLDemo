#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/29 下午12:55 
# @Author : ComeOnJian 
# @File : submit.py

import util
import models
import os
from data import *

import time

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# config
config = util.read_config('./configs/predict.yaml')
torch.manual_seed(config.seed)
# log
log_path = './data/results/'
# log_path = config.log  + '2018-10-20-12:17:22/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = util.logging(log_path+'res_log.txt') # 记录本次运行的记录

# checkpoint
if config.checkpoint_restore:
    print('loading checkpoint from {}...'.format(config.checkpoint_restore))
    checkpoints = torch.load(config.checkpoint_restore)

# cuda
use_cuda = torch.cuda.is_available() and len(config.gpus) > 0
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(config.gpus[0])
    torch.cuda.manual_seed(config.seed)
print('can use cuda: {}'.format(use_cuda))

# data
print('loading data...')
start_time = time.time()
test_dataset = torch.load(config.validation_test_path)
# test_dataset1 = torch.load(config.test_ds_path)
vocab = torch.load(config.vocab_path)
print('loading time cost: %.3f' % (time.time()-start_time))
config.src_vocab_size = vocab.size()
config.tgt_vocab_size = vocab.size()
print('src_vocab_size is {}, and tgt_vocab_size is {}'.format(config.src_vocab_size, config.tgt_vocab_size))
# dataset, batch_size, shuffle, num_workers, mode='train'
test_loader = get_loader(test_dataset, batch_size= config.beam_size, shuffle = config.shuffle, num_workers = config.num_workers, mode='beam_decoder')
# test_loader1 = get_loader(test_dataset1, batch_size= config.beam_size, shuffle = config.shuffle, num_workers = config.num_workers, mode='beam_decoder')

# model
pretrain_emb = {
    'src_emb': None,
    'tgt_emb': None
}

model = getattr(models, config.model)(config, use_cuda, pretrain=pretrain_emb)
if config.checkpoint_restore:
    # 继续训练
    # model.load_state_dict(checkpoints['model'])
    model_dict = model.state_dict()
    pretrain_model_dict = checkpoints['model'].items()
    model_train_dict = {}
    for key, value in pretrain_model_dict:
        if key in model_dict:
            model_train_dict[key] = value
    model_dict.update(model_train_dict)
    model.load_state_dict(model_dict)
if use_cuda:
    model = model.cuda()
if len(config.gpus) > 1:
    model = nn.DataParallel(model, device_ids=config.gpus, dim=0)

def predict(model, test_loader):
    model.eval()
    candidate = []
    # val_loss = 0.
    for idx, batch in enumerate(test_loader):
        sources, source_padding_mask, sources_lengths, sources_batch_extend_vocab, \
        extra_zeros, coverage = get_input_from_batch(batch, use_cuda, use_point_gen=config.pointer_gen,
                                                            use_coverage=config.is_coverage, trian=False, test=True)
        output_ids = model.beam_sample(sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, config.beam_size)
        decoder_words = outputids2words(output_ids, vocab, (batch.art_oovs[0] if config.pointer_gen else None))
        # 去除</s>
        try:
            fst_stop_idx = decoder_words.index('</s>')
            decoder_words = decoder_words[:fst_stop_idx]
        except ValueError:
            decoder_words = decoder_words
        candidate.append(decoder_words)
    util.write_results(candidate)
    model.train()

if __name__ == '__main__':
    predict(model, test_loader)
    # eval()
