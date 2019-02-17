#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/24 下午4:30 
# @Author : ComeOnJian 
# @File : data.py

from gensim.models import KeyedVectors
import torch


PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'  # padding word
UNK_WORD = '<unk>'  # unknow word
BOS_WORD = '<s>'    # target start word
EOS_WORD = '</s>' # target end word

class Vocab(object):

    def __init__(self, vocab_nfile, max_size=None):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

            # Read the vocab file and add words up to max_size
        with open(vocab_nfile, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in {PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD}:
                    print('<s>, </s>, <unk>, <blank> shouldn\'t be in the vocab file, but %s is' % w)
                    continue
                if w in self._word_to_id:
                    print('Duplicated word in vocabulary file: %s' % w)
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNK_WORD]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def build_vectors(self, pre_word_embedding_path, dim , unk_init=torch.Tensor.zero_):
        # unk_init=init.xavier_uniform
        print('load w2v vector from {}...'.format(pre_word_embedding_path))
        pre_word_embedding = KeyedVectors.load_word2vec_format(pre_word_embedding_path, binary=True)
        self.vectors = torch.Tensor(self.size(), dim)
        unknow_count = 0
        for idx, word in self._id_to_word.items():
            if word in pre_word_embedding.vocab:
                wv_index = pre_word_embedding.vocab[word].index
            else:
                wv_index = None
            if wv_index is not None:
                self.vectors[idx] = torch.Tensor(pre_word_embedding.vectors[wv_index])
            else:
                self.vectors[idx] = unk_init(self.vectors[idx].unsqueeze(0)).view(-1)
                unknow_count += 1

        print('the doc vocab length is %d and unkonw count is %d...' % (self.size(), unknow_count))

def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNK_WORD)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNK_WORD)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)

    return " ".join(words)

def abstract2sents(abstract):
    cur = 0
    sents = []
    try:
        start_p = abstract.index(BOS_WORD, cur)
        end_p = abstract.index(EOS_WORD, start_p + 1)
        cur = end_p + len(EOS_WORD)
        sents.append(abstract[start_p+len(BOS_WORD):end_p])
    except ValueError as e: # no more sentences
        return sents

def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNK_WORD)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str

def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNK_WORD)
    words = abstract.split(' ')
    new_words = []
    for w in words:
      if vocab.word2id(w) == unk_token: # w is oov
        if article_oovs is None: # baseline mode
          new_words.append("__%s__" % w)
        else: # pointer-generator mode
          if w in article_oovs:
            new_words.append("__%s__" % w)
          else:
            new_words.append("!!__%s__!!" % w)
      else: # w is in-vocab word
        new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


