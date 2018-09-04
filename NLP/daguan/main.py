#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/8/29 下午4:29 
# @Author : ComeOnJian 
# @File : main.py 
import pandas as pd
import time
import data_analy
import model
import torch.nn as nn
import torch
from torch.autograd import Variable
import lr_scheduler as L
from optims import Optim
from sklearn import metrics
import collections

# load config
config = data_analy.read_config('./config.yaml')
torch.manual_seed(config.seed)

if config.restore:
    print('loading checkpoint...\n')
    check_point = torch.load(config.restore)

# cuda
use_cuda = torch.cuda.is_available() and len(config.gpus) > 0
if use_cuda:
    torch.cuda.set_device(config.gpus[0])
    torch.cuda.manual_seed(config.seed)

# load data
if config.data:
    x_train_articles_ids = data_analy.load(config.x_train_articles_ids)
    x_train_words_ids = data_analy.load(config.x_train_words_ids)
    x_test_articles_ids = data_analy.load(config.x_test_articles_ids)
    x_test_words_ids = data_analy.load(config.x_test_words_ids)
    y_train = data_analy.load(config.y_train)
    y_test = data_analy.load(config.y_test)

    train_dataset = data_analy.dataset(x_train_articles_ids, x_train_words_ids, y_train)
    val_dataset = data_analy.dataset(x_test_articles_ids, x_test_words_ids, y_test)
    dataset = {
        'train': train_dataset,
        'val': val_dataset
    }
    torch.save(dataset,config.data)

else:
    train_dataset = torch.load('./data/dataset')['train']
    val_dataset = torch.load('./data/dataset')['val']

train_dataloader = data_analy.get_loader(train_dataset,
                                         batch_size = config.batch_size,
                                         shuffle = True,
                                         num_workers = 2
                                         )
val_dataloader = data_analy.get_loader(val_dataset,
                                         batch_size = config.batch_size,
                                         shuffle = False,
                                         num_workers = 2
                                         )

# load model
# class_weight = Variable(torch.FloatTensor([1.54, 2.86, 1, 2.17, 3.5, 1.2, 2.73, 1.19, 1.08, 1.67, 2.32, 1.56, 1.05, 1.23, 1.1, 2.58, 2.69, 1.18, 1.5])) # 样本少的类别，可以考虑把权重设置大一点
loss_fn = nn.CrossEntropyLoss(size_average=True)#,weight=class_weight)
if use_cuda:
    loss_fn.cuda()
word_filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
word_filter_nums = [100, 150, 150, 150, 150, 50, 50, 50]
char_filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9]
char_filter_nums = [50, 100, 100, 100, 50, 50, 50, 50]
cnn_model = getattr(model,"Text_WCCNN")(config, word_filter_sizes, word_filter_nums, config.word_vocab_size,
                         char_filter_sizes, char_filter_nums, config.char_vocab_size)

if config.restore:
    cnn_model.load_state_dict(check_point['model'])
if use_cuda:
    cnn_model = cnn_model.cuda()
if len(config.gpus) > 1:
    model = nn.DataParallel(model, device_ids=config.gpus, dim=0)
# optimizer
if config.restore:
    optim = check_point['optim']
    updates = check_point['updates']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    updates = 0
optim.set_parameters(cnn_model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# total number of parameters
param_count = 0
for param in cnn_model.parameters():
    param_count += param.view(-1).size()[0]
print('model all parameters is %d'% param_count)


total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))
# train model
def train(epoch):
    global e
    e = epoch
    cnn_model.train()
    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
    global updates, total_loss, start_time, report_total, report_correct

    for art_src, word_src, y in train_dataloader:

        art_src = Variable(art_src)
        word_src = Variable(word_src)
        y = Variable(y)
        if use_cuda:
            art_src = art_src.cuda()
            word_src = word_src.cuda()
            y = y.cuda()

        cnn_model.zero_grad()
        pre_out = cnn_model(art_src, word_src)
        loss = loss_fn(pre_out,y)
        loss.backward()
        optim.step()
        pred_label = torch.max(pre_out,1)[1]  # Variable
        report_correct += (pred_label.data == y.data).sum()
        report_total += len(y)
        total_loss += loss.data[0]

        updates += 1
        if updates % config.eval_interval == 0:
            print("train--> time: %6.4f, epoch: %3d, updates: %8d, train loss: %6.3f and accuracy: %.3f\n"
                    % ((time.time() - start_time), epoch, updates, total_loss / config.eval_interval,
                       (report_correct / float(report_total))))
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)
            for metric in config.metric:
                scores[metric].append(score[metric])
                if metric == 'macro_f1' and score[metric] >= max(scores[metric]):
                    save_model('./data/' + 'best_' + metric + '_checkpoint.pt')
                if metric == 'loss' and score[metric] <= min(scores[metric]):
                    save_model('./data/' + 'best_' + metric + '_checkpoint.pt')

            cnn_model.train()
            start_time = time.time()
            report_total = 0
            report_correct = 0
            total_loss = 0

        if updates % config.save_interval == 0:
            save_model('./data/model.pt')
        # y:[10] -> [0,0,0,...,1,0,0]
        # y = [to_categorical(item, num_classes) for item in y]

# eval model
def eval(epoch):
    cnn_model.eval()
    y_true = []
    y_pred = []
    eval_total_loss = 0.
    eval_update = 0
    for art_src, word_src, y in val_dataloader:
        art_src = Variable(art_src)
        word_src = Variable(word_src)
        y_true += [y_item for y_item in y]
        y = Variable(y)
        eval_update += 1
        if use_cuda:
            art_src = art_src.cuda()
            word_src = word_src.cuda()
            y = y.cuda()
        if len(config.gpus) > 1:
            output_pro = cnn_model.module.forward(art_src,word_src) # FloatTensor [batch_size,1]
        else:
            output_pro = cnn_model.forward(art_src,word_src)

        loss = loss_fn(output_pro, y)
        eval_total_loss += loss.data[0]
        # y_pred += [item.tolist() for item in output_pro.data] # item is FloatTensor size 19
        pred_label = torch.max(output_pro, 1)[1].data.tolist() # LongTensor size 32
        y_pred += [int(item) for item in pred_label]

    score = {}
    result = get_metrics(y_true,y_pred)
    loss = eval_total_loss/eval_update
    # logging_csv([e, updates, result['f1'], \
    #             result['precision'], result['recall'], loss, result['accuracy']])
    print('eval--> f1: %.4f |precision: %.4f |recall: %.4f |loss: %.4f |accuracy: %.3f '
          % (result['macro_f1'], result['macro_precision'],result['macro_recall'], loss, result['accuracy']))
    score['macro_f1'] = result['macro_f1']
    score['accuracy'] = result['accuracy']
    score['macro_precision'] = result['macro_precision']
    score['macro_recall'] = result['macro_recall']
    score['loss'] = loss
    # score['loss'] = loss

    return score

def save_model(path):
    global updates
    model_state_dict = cnn_model.module.state_dict() if len(config.gpus) > 1 else cnn_model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def get_metrics(y,y_pre):
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    # micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    # micro_precision = metrics.precision_score(y, y_pre, average='micro')
    # micro_recall = metrics.recall_score(y, y_pre, average='micro')
    accuracy = metrics.accuracy_score(y, y_pre)
    result = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }
    return result

if __name__ == '__main__':
    for i in range(config.epoch):
        train(i)
    pass
    # x_train_articles_ids = data_analy.load(config.x_train_articles_ids)
    # # x_train_words_ids = data_analy.load(config.x_train_words_ids)
    # # x_test_articles_ids = data_analy.load(config.x_test_articles_ids)
    # # x_test_words_ids = data_analy.load(config.x_test_words_ids)
    # none_count = 0
    # for article in x_train_articles_ids:
    #     if len(article) == 0:
    #         none_count += 1
    # print(none_count)