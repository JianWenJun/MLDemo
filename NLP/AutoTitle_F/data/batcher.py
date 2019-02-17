#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/24 下午4:52
# @Author : ComeOnJian
# @File : batcher.py

import data
import sys
sys.path.append('../')
import util
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch
import torch.utils.data as torch_data
from torch.autograd import Variable

class Example(object):

    def __init__(self, article, abstract_sentence, vocab, config):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.BOS_WORD)
        stop_decoding = vocab.word2id(data.EOS_WORD)

        # Process the article
        # if article == 'nan':
        #     article_words = ['']
        # else:
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]
        self.enc_len = len(article_words) # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token

        # Process the abstract
        abstract_words = abstract_sentence.split() # list of strings
        abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

        # Store the original strings
        # self.original_article = article
        self.original_abstract = abstract_sentence

    def get_dec_inp_seqs(self, sequence, max_len, start_id, stop_id):
        dec_input = [start_id] + sequence[:]
        target = sequence[:]
        if len(dec_input)>max_len:
            dec_input = dec_input[:max_len]
            target = target[:max_len]
        else:
            target.append(stop_id)
        return dec_input, target

    def pad_decoder_inp(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id, pointer_gen=True):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

class Batch(object):
    def __init__(self, example_list, batch_size):
        self.batch_size = batch_size
        self.pad_id = data.PAD  # id of the PAD token used to pad sequences
        # 根据example_list的enc_len长度进行降序排列
        example_list.sort(key=lambda ex: ex.enc_len, reverse=True)
        self.init_encoder_seq(example_list) # initialize the input to the encoder
        self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings

    def init_encoder_seq(self, example_list ,pointer_gen = True):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
        # For pointer-generator mode, need to store some extra info

        if pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        max_dec_seq_len = max([len(ex.dec_input) for ex in example_list])

        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp(max_dec_seq_len, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, max_dec_seq_len), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, max_dec_seq_len), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]

    def store_orig_strings(self, example_list):
        # self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists

################### construct model input ###################

class DocDataset(torch_data.Dataset):

    def __init__(self, path, vocab, config):
        df_data = pd.read_csv(path, sep='\t', header=None)
        if config.test:
            df_data.columns = ['id', 'content']
        else:
            df_data.columns = ['id', 'content', 'title']
        print(df_data.shape)
        self.samples = []
        for source, target, i in tqdm(zip(df_data['content'], df_data['title'], df_data['id'])):
            if config.aug:
                # 数据增强
                rate = random.random()
                if rate > 0.5:
                    source = self.drpout(source)
                else:
                    source = self.shuffle(source)
            try:
                self.samples.append(Example(str(source), target, vocab, config))
            except Exception:
                print('setence id: {} has problem,--{}'.format(i, target))
                continue

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def drpout(self,text, p = 0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_*p))
        for i in indexs:
            text[i] = ''
        return " ".join(text)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

def padding(example_list):
    batch_size = len(example_list)
    # <s> + </s> + 20 = 22
    # decoder模式下:
    batch = Batch(example_list, batch_size)
    return batch

def get_loader(dataset, batch_size, shuffle, num_workers, mode='train'):
    if mode =='beam_decoder':
        # 每个样本重复batch_size次
        samples = []
        for i in range(len(dataset)):
            for j in range(batch_size):
                samples.append(dataset[i])
        dataset.samples = samples
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader

def get_input_from_batch(batch, use_cuda, use_point_gen = True, use_coverage = False, trian = True, test=False):
    ##################### encoder ######################
    batch_size = batch.batch_size
    source = Variable(torch.from_numpy(batch.enc_batch).long(),volatile=not trian)
    source_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask).float(),volatile=not trian)
    source_lengths = torch.from_numpy(batch.enc_lens) # numpy array
    extra_zeros = None
    source_batch_extend_vocab = None

    if use_point_gen:
        source_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long(),volatile=not trian)
        if batch.max_art_oovs>0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)),volatile=not trian)
    coverage = None
    if use_coverage:
        coverage = Variable(torch.zeros(source.size()))

    if use_cuda:
        source = source.cuda()
        source_padding_mask = source_padding_mask.cuda()
        if source_batch_extend_vocab is not None:
            source_batch_extend_vocab = source_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if coverage is not None:
            coverage = coverage.cuda()
    if test:
        return source, source_padding_mask, source_lengths, source_batch_extend_vocab, extra_zeros, coverage
    ##################### decoder ######################
    encoder_inputs = Variable(torch.from_numpy(batch.dec_batch).long(), volatile=not trian)  # 有<s> pad ,oov没有word_id,为unk
    target = Variable(torch.from_numpy(batch.target_batch).long(), volatile=not trian)  # 有</s> pad，oov有word_id
    target_raw = batch.original_abstracts  # list
    if use_cuda:
        target = target.cuda()
        encoder_inputs = encoder_inputs.cuda()
    return source, source_padding_mask, source_lengths, source_batch_extend_vocab, extra_zeros, coverage, \
           encoder_inputs, target, target_raw

def get_temp_vocab(config):
    from collections import Counter
    df_data1 = pd.read_csv(config.train_path, sep='\t', header=None)
    df_data1.columns = ['id', 'content', 'title']
    df_data2 = pd.read_csv(config.val_path, sep='\t', header=None)
    df_data2.columns = ['id', 'content', 'title']
    df_data3 = pd.read_csv(config.test_path, sep='\t', header=None)
    df_data3.columns = ['id', 'content', 'title']
    df_all = pd.concat([df_data1, df_data2, df_data3], ignore_index=True)

    print(df_all.shape)
    print('finish read...')
    vocab_counter = Counter()
    for index, row in df_all.iterrows():
        title_words = str(row['title']).lower().split()
        content_words = str(row['content']).lower().split()
        vocab_counter.update(title_words)
        vocab_counter.update(content_words)
        df_all.at[index, 'title'] = " ".join(title_words)
        df_all.at[index, 'content'] = " ".join(content_words)
    print("Writing vocab file...")
    with open(config.vocab_path, 'w') as writer:
        for word, count in vocab_counter.most_common(200000):
            writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")
    df_all[:-2000].to_csv('../data/preprocessed/train_set_last_all.csv', sep='\t', header=None, index=None, encoding='utf-8')
    df_all[-2000:-1000].to_csv('../data/preprocessed/val_set_last_1000.csv', sep='\t', header=None, index=None, encoding='utf-8')
    df_all[-1000:].to_csv('../data/preprocessed/test_set_last_1000.csv', sep='\t', header=None, index=None, encoding='utf-8')

def build_vaildation_set():
    class Config:
        pass
    config = Config()
    setattr(config,'test', False)
    setattr(config,'aug', False)
    setattr(config,'save', True)
    setattr(config,'max_enc_steps', 600)
    setattr(config,'max_dec_steps', 20)
    setattr(config,'pointer_gen', True)
    setattr(config,'coverage', False)
    vocab_path = './preprocessed/dataset_50k_600/vocab_50000'
    vaildation_path = './results/test_set_add_title.csv'
    vocab = torch.load(vocab_path)
    vaildation_dataset = DocDataset(vaildation_path, vocab, config)
    if config.save:
        torch.save(vaildation_dataset, './results/test.data')

def main():
    from torch.nn import init
    config = util.read_config('../configs/process.yaml')
    # get_temp_vocab(config)
    vocab = data.Vocab(config.vocab_path, max_size=config.max_size)
    # vocab.build_vectors(config.pre_word_embedding_path, 300, unk_init=init.xavier_uniform)
    if config.save:
        torch.save(vocab, config.vocab_path_50)
    val_data = DocDataset(config.val_path, vocab, config)
    test_data = DocDataset(config.test_path, vocab, config)
    if config.save:
        torch.save(val_data, config.val_data_path)
        torch.save(test_data, config.test_data_path)

    train_data = DocDataset(config.train_path, vocab, config)
    if config.save:
        torch.save(train_data, config.train_data_path)

if __name__ == '__main__':
    # main()
    build_vaildation_set()
    pass



