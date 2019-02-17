#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/25 下午3:45 
# @Author : ComeOnJian 
# @File : train.py 

import util
import models
import os
import time
from collections import OrderedDict
from data import *

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# config
config = util.read_config('./configs/train_model.yaml')
torch.manual_seed(config.seed)

# log
log_path = config.log + util.format_time(time.localtime()) + '/'
# log_path = config.log  + '2018-10-20-12:17:22/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = util.logging(log_path+'log.txt') # 记录本次运行的记录
logging_csv = util.logging_csv(log_path+'record.csv') # 记录模型训练的指标数据

# checkpoint
if config.checkpoint_restore:
    print('loading checkpoint from {}...'.format(config.checkpoint_restore))
    # map_location={'cuda:1':'cuda:0'}
    checkpoints = torch.load(config.checkpoint_restore)

# cuda
use_cuda = torch.cuda.is_available() and len(config.gpus) > 0
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(config.gpus[0])
    torch.cuda.manual_seed(config.seed + 1)
logging('can use cuda: {}'.format(use_cuda))

# data
logging('loading data...')
start_time = time.time()
train_dataset = torch.load(config.train_ds_path)
val_dataset = torch.load(config.val_ds_path)
vocab = torch.load(config.vocab_path)
logging('loading time cost: %.3f' % (time.time()-start_time))
config.src_vocab_size = vocab.size()
config.tgt_vocab_size = vocab.size()
logging('src_vocab_size is {}, and tgt_vocab_size is {}'.format(config.src_vocab_size, config.tgt_vocab_size))
# dataset, batch_size, shuffle, num_workers, mode='train'
train_loader = get_loader(train_dataset, batch_size = config.batch_size, shuffle = not config.shuffle, num_workers = config.num_workers)
val_loader = get_loader(val_dataset, batch_size= config.beam_size, shuffle = config.shuffle, num_workers = config.num_workers-1,mode='beam_decoder')

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
# optim
if config.checkpoint_restore:
    optim = checkpoints['optim']
else:
    # optim = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
    #               lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optim = models.Optim(config.optim, config.learning_rate, config.max_grad_norm, initial_accumulator_value=config.adagrad_init_acc)
optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = models.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]
logging('model have {} param'.format(param_count))
for k, v in config.items():
    logging("%s:\t%s" % (str(k), str(v)))
logging(repr(model)+"\n")

# model relation data
if config.checkpoint_restore:
    updates = checkpoints['updates']
else:
    updates = 0
# 模型统计指标
total_loss, start_time = 0., time.time()
report_total = 0
scores = [[] for metric in config.metrics]
scores = OrderedDict(zip(config.metrics, scores))

def train(epoch):
    global e
    e = epoch
    model.train()
    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
    global updates, start_time, total_loss
    for idx, batch in enumerate(train_loader):
        sources, source_padding_mask, sources_lengths, sources_batch_extend_vocab, extra_zeros, coverage, \
        encoder_inputs, targets, _ = get_input_from_batch(batch, use_cuda, use_point_gen = config.pointer_gen,
                                        use_coverage = config.is_coverage, trian = True)

        model.zero_grad()
        mean_loss = model(sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, encoder_inputs, targets)
        total_loss += mean_loss
        optim.step()
        updates += 1
        # eval
        if updates % config.eval_interval == 0:
            score = eval(epoch)
            for metric in config.metrics:
                scores[metric].append(score[metric])
                # if metric == 'loss' and score[metric] <= min(scores[metric]):
                #     logging('{} has updated, {} updates turn out min value: {}'.format(metric, updates, score[metric]))
                #     save_model(log_path + 'best_' + metric + '_checkpoint.pt')
                if metric == 'rouge_l' and score[metric] >= max(scores[metric]):
                    logging('{} has updated, {} updates turn out max value: {}'.format(metric, updates, score[metric]))
                    save_model(log_path + 'best_' + metric + '_checkpoint.pt')
                if metric == 'bleu_2' and score[metric] >= max(scores[metric]):
                    logging('{} has updated, {} updates turn out max value: {}'.format(metric, updates, score[metric]))
                    save_model(log_path + 'best_' + metric + '_checkpoint.pt')
            model.train()


        # print train result
        if updates % config.print_interval == 0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train_loss: %6.3f"
                    % (time.time() - start_time, epoch, updates, total_loss / config.print_interval))
            total_loss = 0
            start_time = time.time()

        # save model
        if updates % config.save_interval == 0:
            save_model(log_path + '{}.checkpoint.pt'.format(updates))
        torch.cuda.empty_cache()

def eval(epoch):
    model.eval()
    reference, candidate = [], []
    # val_loss = 0.
    for idx, batch in enumerate(val_loader):
        sources, source_padding_mask, sources_lengths, sources_batch_extend_vocab, extra_zeros, coverage, \
        _,  _, targets_raw = get_input_from_batch(batch, use_cuda, use_point_gen=config.pointer_gen,
                                                            use_coverage=config.is_coverage, trian=False)

        # if updates%2000==0:
        output_ids = model.beam_sample(sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, config.beam_size)
        # else:
            # _, sample_y = model.rein_forward(sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, tag_max_l, mode = 'sample_max')
        reference.append(targets_raw[0])
        decoder_words = outputids2words(output_ids, vocab, (batch.art_oovs[0] if config.pointer_gen else None))
        # 去除</s>
        try:
            fst_stop_idx = decoder_words.index('</s>')
            decoder_words = decoder_words[:fst_stop_idx]
        except ValueError:
            decoder_words = decoder_words
        candidate.append(decoder_words)

    score = {}
    result = util.eval_metrics(candidate, reference)
    # result['val_loss'] = val_loss result['val_loss'], (result['val_loss']
    logging_csv([e, updates, result['bleu_1'], result['bleu_2'], result['rouge_l'], result['cider']])
    print('eval: |bleu_1: %.4f |bleu_2: %.4f |rouge_l: %.4f |cider: %.4f'
          % (result['bleu_1'], result['bleu_2'], result['rouge_l'],result['cider']))
    # score['val_loss'] = result['val_loss']
    score['rouge_l'] = result['rouge_l']
    score['bleu_1'] = result['bleu_1']
    score['bleu_2'] = result['bleu_2']
    score['cider'] = result['cider']
    del reference, candidate, sources, source_padding_mask, sources_lengths, sources_batch_extend_vocab, extra_zeros, coverage, targets_raw
    torch.cuda.empty_cache()
    return score

def main():
    for i in range(6, config.epoch+1):
        if i < 7:
            config.eval_interval = 5000
            config.save_interval = 5000
        elif i>= 7:
            config.eval_interval = 200
            config.save_interval = 1500
        train(i)

    for metric in config.metrics:
        if metric in [ 'bleu_1', 'bleu_2', 'rouge_l']:
            logging("Best %s score: %.2f" % (metric, max(scores[metric])))
        else:
            logging("Best %s score: %.2f" % (metric, min(scores[metric])))

def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(config.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)

if __name__ == '__main__':
    # try:
    main()
    # except Exception:
    #     if model is not None:
    #         save_model(log_path + 'last_{}.checkpoint.pt'.format(updates))
    #     exit()
