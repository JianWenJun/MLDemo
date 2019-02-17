#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 下午5:10 
# @Author : ComeOnJian 
# @File : loss.py 
import torch
import torch.nn as nn
from torch.autograd import Variable
import models.adaptive as adaptive
PAD = 0
############   ###################
def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[PAD] = 0
    # 如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss
    # 如果 reduce = True，那么 loss 返回的是标量。
    # 如果 size_average = True，返回 loss.mean(). 如果 size_average = False，返回 loss.sum();
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

def adaptive_criterion(config, use_cuda):
    #  适合切5-10份
    # splits = [2800, 20000, config.tgt_vocab_size - 1]  # 中型vocab
    # splits = [4200, 35000, 180000] # 大型vocab
    crit = adaptive.AdaptiveLogSoftmaxWithLoss(in_features=config.encoder_hidder_size,
                                                    n_classes=config.tgt_vocab_size,
                                                    cutoffs=config.splits)
    if use_cuda:
        crit.cuda()
    return crit

def ml_criterion(hidden_outputs, decoder, targets, criterion, sim_score=0):
    """maximum likelihood loss"""
    # targets # [seq_length, batch]
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2)) # [seq_length,batch,encoder_hidder_size] -> [seq_length * batch, encoder_hidder_size]
    scores = decoder.compute_score(outputs) # [seq_length * batch, encoder_hidder_size] -> [seq_length * batch, tgt_vocab_size]
    loss = criterion(scores, targets.view(-1)) + sim_score # [1]
    # pred = scores.max(1)[1] # [seq_length * batch, tgt_vocab_size] -> [seq_length * batch]
    # num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD).data).sum()
    num_total = targets.ne(PAD).data.sum()
    loss.div(num_total).backward()
    loss = loss.data[0]

    del outputs, scores
    torch.cuda.empty_cache()
    return loss, num_total #, num_correct

def ml_criterion_memory_efficiency(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, loss = 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        num_total_t = targ_t.ne(PAD).data.sum()
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)
    return loss, num_total

def ml_criterion_sampled_loss(hidden_outputs, decoder, targets, config, sim_score=0):

    pass

def ml_criterion_adaptive_sampled_loss(hidden_outputs, decoder, targets, criterion):

    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))  # [seq_length,batch,encoder_hidder_size] -> [seq_length * batch, encoder_hidder_size]
    scores = decoder.compute_score(outputs)  # [seq_length * batch, encoder_hidder_size] -> [seq_length * batch, tgt_vocab_size]
    raw_loss = criterion(scores, targets.view(-1))  # [1]
    # pred = scores.max(1)[1] # [seq_length * batch, tgt_vocab_size] -> [seq_length * batch]
    # num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD).data).sum()
    num_total = targets.ne(PAD).data.sum()
    raw_loss[1].div(num_total).backward()
    loss = raw_loss[1].data[0]

    del outputs, scores
    torch.cuda.empty_cache()
    return loss, num_total, raw_loss# , num_correct

def rl_criterion():
    pass

