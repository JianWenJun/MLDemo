#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 上午10:50 
# @Author : ComeOnJian 
# @File : seq2seq.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
nn.SELU
PAD = 0
UNK = 1
BOS = 2
EOS = 3

# init param
def init_lstm_wt(lstm, init_v):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-init_v, init_v)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear, init_v):
    # init_v = config.trunc_norm_init_std
    linear.weight.data.normal_(std=init_v)
    if linear.bias is not None:
        linear.bias.data.normal_(std=init_v)

def init_wt_normal(wt, init_v):
    wt.data.normal_(std=init_v)

def init_wt_unif(wt, init_v):
    # init_v = config.trunc_norm_init_std
    wt.data.uniform_(-init_v, init_v)

## encoder
class Encoder(nn.Module):
    def __init__(self, config, embedding_weight = None):
        super(Encoder, self).__init__()
        # word embedding
        self.embedding = nn.Embedding(config.src_vocab_size, config.emb_dim)
        if embedding_weight is None:
            init_wt_normal(self.embedding.weight, config.trunc_norm_init_std)
        else:
            self.embedding.weight.data.copy_(embedding_weight)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=config.encoder_num_layers, batch_first=True, bidirectional=config.encoder_bidirec)
        init_lstm_wt(self.lstm, config.rand_unif_init_mag)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed) # hidden ->(h,c) -> h or c num_layer x B x n

        h, _ = unpack(output, batch_first=True)  # h dim = B x t_k x n
        h = h.contiguous()
        max_h, _ = h.max(dim=1)

        return h, hidden, max_h

class ReduceState(nn.Module):
    def __init__(self,config):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h, config.rand_unif_init_mag)
        self.reduce_c = nn.Linear(config.hidden_dim * 2 , config.hidden_dim)
        init_linear_wt(self.reduce_c, config.rand_unif_init_mag)
        self.config = config

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2 )
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.config.hidden_dim * 2 )
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

## decoder
class Decoder(nn.Module):
    def __init__(self, config, embedding_weight = None):
        super(Decoder, self).__init__()
        self.attention_network = Attention(config)
        # decoder
        self.embedding = nn.Embedding(config.tgt_vocab_size, config.emb_dim)
        if embedding_weight is None:
            init_wt_normal(self.embedding.weight, config.trunc_norm_init_std)
        else:
            self.embedding.weight.data.copy_(embedding_weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm, config.rand_unif_init_mag)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.tgt_vocab_size)
        init_linear_wt(self.out2,config.trunc_norm_init_std)
        self.config = config

    def forward(self, y_t_1, s_t_1, encoder_outputs, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),
                                 c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.config.hidden_dim),
                             c_decoder.view(-1, self.config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

## attention
class Attention(nn.Module):
    def __init__(self,config):
        super(Attention, self).__init__()
        # attention
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2 , config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)
        self.config = config

    def forward(self, s_t_hat, h, enc_padding_mask, coverage):
        b, t_k, n = list(h.size())
        h = h.view(-1, n)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(h) #b

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if self.config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.view(-1, t_k, n)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, self.config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

#包含全局info self attention + cnn
class Global_Attention(nn.Module):
    pass


## beam sample
class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)

# seq2seq
class seq2seq(nn.Module):
    def __init__(self, config, use_cuda, pretrain = None):
        super(seq2seq, self).__init__()
        if pretrain is not None:
            src_embedding = pretrain['src_emb']
            tgt_embedding = pretrain['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None

        self.encoder = Encoder(config, embedding_weight=src_embedding)
        self.decoder = Decoder(config, embedding_weight=tgt_embedding)
        self.reduce_state = ReduceState(config)
        if config.share_vocab:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        self.use_cuda = use_cuda
        self.config = config

    def forward(self, sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, encoder_inputs, targets):
        encoder_outputs, encoder_hidden, max_encoder_output = self.encoder(sources, sources_lengths.tolist())
        s_t_1 = self.reduce_state(encoder_hidden)
        size = len(sources_lengths.tolist())
        c_t_1 = Variable(torch.zeros((size, 2 * self.config.hidden_dim)),volatile= False)
        if self.config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output
        if self.config.use_cuda:
            c_t_1 = c_t_1.cuda()
        if self.config.is_semantic_similary:
            self.en_h = s_t_1[0].squeeze(0) # b x hidden_dim
        step_losses = []
        max_l = targets.size()[1]
        for di in range(max_l):
            y_t_1 = encoder_inputs[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           source_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           sources_batch_extend_vocab,
                                                                                           coverage, di)
            target = targets[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.config.eps)
            if self.config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            step_mask = target.ne(PAD).float()
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        step_losses = torch.stack(step_losses, 1)
        num_total = targets.data.ne(PAD).sum()
        mean_loss = torch.sum(step_losses) / num_total
        if self.config.is_semantic_similary:
            self.de_h = s_t_1[0].squeeze(0) - self.en_h
            mean_loss = mean_loss - self.config.simil_wt * F.cosine_similarity(self.en_h,self.de_h,eps=self.config.eps)
        mean_loss.backward()
        return mean_loss.data[0]

    def rein_forward(self, sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros,
                    coverage, tag_max_l, mode = 'sample'):
        # mode = 'sample_max' or 'sample'
        if mode != 'sample' and mode != 'sample_max':
            raise ValueError('mode param must is sample or sample_max')
        encoder_outputs, encoder_hidden, max_encoder_output = self.encoder(sources, sources_lengths.tolist())
        s_t_1 = self.reduce_state(encoder_hidden)
        size = len(sources_lengths.tolist())
        c_t_1 = Variable(torch.zeros((size, 2 * self.config.hidden_dim)), volatile=(mode=='sample_max'))
        if self.config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output
        y_t_1 = Variable(torch.LongTensor(size).fill_(BOS), volatile=(mode=='sample_max'))
        if self.config.use_cuda:
            c_t_1 = c_t_1.cuda()
            y_t_1 = y_t_1.cuda()
        # 不用teach forcing，根据采样分布sample选取
        sample_y = []
        log_probs = []
        for di in range(tag_max_l):
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                     encoder_outputs,
                                                                                     source_padding_mask, c_t_1,
                                                                                     extra_zeros,
                                                                                     sources_batch_extend_vocab,
                                                                                     coverage, di)
            if mode == 'sample':
                y_t_1 = torch.multinomial(final_dist, 1).view(-1)  # word_id可能oov,需要将 oov 换成unk
                gold_probs = torch.gather(final_dist, 1, y_t_1.unsqueeze(1)).squeeze()

            elif mode == 'sample_max':
                gold_probs, y_t_1 = torch.max(final_dist, 1)
            sample_y.append(y_t_1)
            y_t_1 = self.variable_to_init_id(y_t_1)
            step_loss = -torch.log(gold_probs + self.config.eps)
            if self.config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
            step_mask = sample_y[-1].ne(PAD).float()
            step_loss = step_loss * step_mask
            log_probs.append(step_loss)
        log_probs = torch.stack(log_probs, 1)
        sample_y = torch.stack(sample_y, 1)
        return log_probs, sample_y  # [B, max_seq_l]

    def beam_sample(self, sources, sources_lengths, source_padding_mask, sources_batch_extend_vocab, extra_zeros, coverage, beam_size):
        encoder_outputs, encoder_hidden, max_encoder_output = self.encoder(sources, sources_lengths.tolist())
        b = max_encoder_output.size()[0]
        s_t_0 = self.reduce_state(encoder_hidden)
        c_t_0 = Variable(torch.zeros((b, 2 * self.config.hidden_dim)), volatile=False)
        if self.config.use_maxpool_init_ctx:
            c_t_0 = max_encoder_output
        if self.config.use_cuda:
            c_t_0 = c_t_0.cuda()
        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze(dim=0) #默认会把前两个都squeeze了
        dec_c = dec_c.squeeze(dim=0)
        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[BOS],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage[0] if self.config.is_coverage else None))
                 for _ in range(beam_size)]
        results = []
        steps = 0
        while steps < self.config.max_dec_steps and len(results) < beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.config.tgt_vocab_size else UNK for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens),volatile=False)
            if self.use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, source_padding_mask, c_t_1,
                                                        extra_zeros, sources_batch_extend_vocab, coverage_t_1, steps)

            topk_log_probs, topk_ids = torch.topk(final_dist, beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if self.config.is_coverage else None)

                for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].data[0],
                                   log_prob=topk_log_probs[i, j].data[0],
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == EOS:
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == beam_size or len(results) == beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        output_ids = [int(t) for t in beams_sorted[0].tokens[1:]]
        return output_ids

    def sort_beams(self, beams):
        beams.sort(key=lambda h: h.avg_log_prob, reverse=True)
        return beams

    def variable_to_init_id(self, v):
        li = [t if t < self.config.tgt_vocab_size else UNK for t in v.data.tolist()]
        v_data = torch.LongTensor(li)
        if self.config.use_cuda:
            v_data = v_data.cuda()
        v.data = v_data
        return v

    # def comput_rewardCritierion(self, inputs,  ):
    #     pass

