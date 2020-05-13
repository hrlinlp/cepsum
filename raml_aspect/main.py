# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py --mode=<mode> [options]

Options:
    -h --help                               show this screen.
    --mode=<mode>                           train or decode
    --cuda=<bool>                           use GPU [default: True]
    --copy=<bool>                           apply copy mechanism [default: True]
    --coverage=<bool>                       apply coverage mechanism [default: False]
    --train-json=<file>                     train json file [default: ../data/cases_bags_train.json]
    --dev-json=<file>                       dev json file [default: ../data/cases_bags_dev.json]
    --test-json=<file>                      test json file [default: ../data/cases_bags_test.json]
    --model-path=<file>                     model file
    --output=<file>                         output file
    --vocab=<file>                          vocab file [default: ../data/vocab.json]
    --img=<file>                            image name file [default: ../data/image_fc_names_images_xiangbao_rgb_idx]
    --img-fc=<file>                         image avepool file [default: ../data/image_fc_vectors_images_xiangbao_rgb.npy]
    --seed=<int>                            seed [default: 5783287]
    --batch-size=<int>                      batch size [default: 64]
    --embed-size=<int>                      embedding size [default: 300]
    --hidden-size=<int>                     hidden size [default: 512]
    --clip-grad=<float>                     gradient clipping [default: 2.0]
    --log-every=<int>                       log every [default: 500]
    --max-epoch=<int>                       max epoch [default: 999]
    --input-feed={bool}                     use input feeding [default: True]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 7]
    --max-num-trial=<int>                   terminate training after how many trials [default: 7]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 10]
    --lr=<float>                            learning rate [default: 0.0005]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 10]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 80]
    --load-model=<str>                      continue training [default: False]
    --num-trial=<int>                       having trained for how many trials [default: 0]
    --best-r2=<float>                       best rouge2 f1 score for the trained models [default: 0.0]
    --best-ppl=<float>                      minimum ppl score for the trained models [default: 1]
    --aspect-file=<file>                    aspect file path [default: ../aspects/]
    --aspect-active=<bool>                  aspect repeat forbidden [default: True]
    --trichar-repeat-active=<bool>          trichar repeat forbidden [default: True]
"""
from __future__ import print_function
import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import read_corpus_json, read_corpus, batch_iter, LabelSmoothingLoss, read_aspect_file, read_test_corpus, read_image, read_corpus
from vocab import Vocab, VocabEntry

from rougescore import *
import copy
import re

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0., copy=False, coverage=False):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.input_feed = input_feed
        self.copy = copy
        self.coverage = coverage

        # initialize neural network layers...

        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        decoder_lstm_input = embed_size + hidden_size if self.input_feed else embed_size
        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's state space
        self.att_src_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)

        self.att_ht_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_v_linear = nn.Linear(hidden_size, 1, bias=False)
        #self.att_c_linear = nn.Linear(1, hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(hidden_size * 2 + 2048, hidden_size)

        if self.copy:
            self.p_linear = nn.Linear(hidden_size * 3 + embed_size, 1)

        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt), padding_idx=vocab.tgt['<pad>'])

    @property
    def device(self):
        return self.src_embed.weight.device

    def forward(self, src_sents, tgt_sents, img_vecs, aspects, aspect_idx_dicts):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (src_sent_len, batch_size); (tgt_sent_len, batch_size)
        src_sents_var = self.vocab.src.to_input_tensor_src(src_sents, device=self.device)
        if self.copy:
            (tgt_sents_var, src_complete_sents_var, tgt_complete_sents_var,
                    word_oovs, max_oov_num) = self.vocab.tgt.to_input_tensor_tgt(src_sents, tgt_sents, device=self.device)
        else: #
            src_sents_var = self.vocab.tgt.to_input_tensor_src(src_sents, device=self.device)
            tgt_sents_var = self.vocab.tgt.to_input_tensor_src(tgt_sents, device=self.device)

        aspect_idxs = []
        for src_sent, aspect in zip(src_sents, aspects):
            aspect_idx_dict = aspect_idx_dicts[aspect]
            aspect_idx = [aspect_idx_dict.get(w, 0) for w in src_sent]
            aspect_idxs.append(aspect_idx)
        max_len = max(len(aspect_idx) for aspect_idx in aspect_idxs)
        batch_size = len(aspect_idxs)
        aspect_t = []
        for k in range(batch_size):
            aspect_t.append([aspect_idxs[k][i] if len(aspect_idxs[k]) > i else 0 for i in range(max_len)])
        aspect_idxs_var = torch.tensor(aspect_t, dtype=torch.long, device=self.device)

        imgs_fc_var = torch.tensor(img_vecs, dtype=torch.float, device=self.device)
        src_sents_len = [len(s) for s in src_sents]

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len, imgs_fc_var)

        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        if self.copy and self.coverage:
            h_ts, ctx_ts, alpha_ts, att_vecs, tgt_word_embeds, coverages = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])
        elif self.coverage:
            att_vecs, coverages = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])
        elif self.copy:
            h_ts, ctx_ts, alpha_ts, att_vecs, tgt_word_embeds, asp_coverages = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1], aspect_idxs_var)
            # h_ts:            (tgt_sent_len - 1, batch_size, hidden_size)
            # ctx_ts:          (tgt_sent_len - 1, batch_size, hidden_size * 2)
            # alpha_ts:        (tgt_sent_len - 1, batch_size, src_sent_len)
            # att_vecs:        (tgt_sent_len - 1, batch_size, hidden_size)
            # tgt_word_embeds: (tgt_sent_len - 1, batch_size, embed-size)
        else:
            att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        if self.copy:
            tgt_words_log_prob = F.softmax(self.readout(att_vecs), dim=-1)

            if max_oov_num > 0:
                oov_zeros = torch.zeros(tgt_words_log_prob.size(0), tgt_words_log_prob.size(1), max_oov_num, device=self.device)
                tgt_words_log_prob = torch.cat([tgt_words_log_prob, oov_zeros], dim=-1)

            p = torch.cat([h_ts, ctx_ts, tgt_word_embeds], dim=-1)
            g = torch.sigmoid(self.p_linear(p))

            g = torch.clamp(g, 1e-9, 1 - 1e-9)

            src_complete_sents_var_expanded = src_complete_sents_var.permute(1, 0).expand(alpha_ts.size(0), -1, -1)
            tgt_words_log_prob = (g * tgt_words_log_prob).scatter_add(2, src_complete_sents_var_expanded, (1 - g) * alpha_ts)


        else:
            tgt_words_log_prob = F.log_softmax(self.readout(att_vecs), dim=-1)

        # (tgt_sent_len, batch_size)
        tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

        # (tgt_sent_len - 1, batch_size)
        if self.copy:
            tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob,
                        index=tgt_complete_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1)
            tgt_gold_words_log_prob = torch.log(tgt_gold_words_log_prob) * tgt_words_mask[1:]
        else:
            tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # (batch_size)
        scores = tgt_gold_words_log_prob.sum(dim=0) / torch.sum(tgt_words_mask[1:], 0)
        coverage_loss = asp_coverages.sum(dim=0).sum(dim=1) / torch.sum(tgt_words_mask[1:], 0)

        if self.coverage:
            return scores, coverages
        else:
            return scores, coverage_loss

    def get_attention_mask(self, src_encodings, src_sents_len):
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(self.device)

    def encode(self, src_sents_var, src_sent_lens, imgs_fc_var):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        # (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)

        # src_encodings: (src_sent_len, batch_size, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1], imgs_fc_var], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)

        return src_encodings, (dec_init_state, dec_init_cell)

    def decode(self, src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var, aspect_idxs_var):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = decoder_init_vec

        att_ves = []

        if self.copy:
            h_ts = []
            ctx_ts = []
            alpha_ts = []
            asp_att_history = None
            asp_coverages = []

        if self.coverage:
            att_history = None
            coverages = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            if self.coverage:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, att_history)
                if att_history is None:
                    att_history = alpha_t
                else:
                    coverage = torch.min(alpha_t, att_history)
                    coverages.append(coverage)
                    att_history = att_history + alpha_t
            else:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, asp_att_history)
                alpha_asp = torch.zeros_like(aspect_idxs_var, dtype=torch.float).scatter_add(1, aspect_idxs_var, alpha_t)

                alpha1 = torch.clone(alpha_asp)
                alpha1[:, 0] = 0
                alpha_asp = alpha1
                alpha_asp_gather = torch.gather(alpha_asp, 1, aspect_idxs_var)
                if asp_att_history is None:
                    asp_att_history = alpha_asp_gather
                else:
                    asp_coverage = torch.min(alpha_asp_gather, asp_att_history)
                    asp_coverages.append(asp_coverage)
                    asp_att_history = asp_att_history + alpha_asp_gather

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)
            if self.copy:
                h_ts.append(h_t)
                ctx_ts.append(ctx_t)
                alpha_ts.append(alpha_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        if self.copy:
            h_ts = torch.stack(h_ts)
            ctx_ts = torch.stack(ctx_ts)
            alpha_ts = torch.stack(alpha_ts)
            asp_coverages = torch.stack(asp_coverages)

        if self.coverage:
            coverages = torch.stack(coverages)

        if self.copy and self.coverage:
            return h_ts, ctx_ts, alpha_ts, att_ves, tgt_word_embeds, coverages
        elif self.coverage:
            return att_ves, coverages
        elif self.copy:
            return h_ts, ctx_ts, alpha_ts, att_ves, tgt_word_embeds, asp_coverages
        else:
            return att_ves

    def step(self, x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, att_history):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks, att_history)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), ctx_t, alpha_t, att_t

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask, att_history):
        # (batch_size, src_sent_len, hidden_size) * (batch_size, hidden_size, 1) = (batch_size, src_sent_len)

        if self.coverage and att_history is not None:
            att_hidden_text = torch.tanh(self.att_ht_linear(h_t).unsqueeze(1).expand_as(src_encoding_att_linear) + att_history.unsqueeze(2).expand_as(src_encoding_att_linear) + src_encoding_att_linear)
        else:
            if att_history is None:
                att_hidden_text = torch.tanh(self.att_ht_linear(h_t).unsqueeze(1).expand_as(src_encoding_att_linear) + src_encoding_att_linear)
            else:
                att_hidden_text = torch.tanh(self.att_ht_linear(h_t).unsqueeze(1).expand_as(src_encoding_att_linear) + self.att_c_linear(att_history.unsqueeze(2)) + src_encoding_att_linear)



        # (batch_size, src_sent_len)
        att_weight = self.att_v_linear(att_hidden_text).squeeze(2)


        if mask is not None:
            att_weight.data.masked_fill_(mask.byte(), -float('inf'))

        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight

    def beam_search(self, src_sent, img, aspect_sent, aspect_idx, first_uni2id, first_bi2id, tri2id, beam_size, max_decoding_time_step,  aspect_map, args):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        aspect_active = str_to_bool(args['--aspect-active'])

        trichar_repeat_active = str_to_bool(args['--trichar-repeat-active'])

        src_sents_var = self.vocab.src.to_input_tensor_src([src_sent], device=self.device)
        imgs_fc_var = torch.tensor([img], dtype=torch.float, device=self.device)
        aspect_idx_var = torch.tensor([aspect_idx], dtype=torch.long, device=self.device)

        if self.copy:
            (src_complete_sents_var, word_oovs, max_oov_num) = self.vocab.tgt.to_input_tensor_tgt_decode([src_sent], device=self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)], imgs_fc_var)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        hypotheses = [['<s>']]
        forbidden_ids = [[]]

        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        if self.coverage:
            att_history = None
        asp_att_history = None
        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            if self.coverage:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear,
                                                                 None, att_history)
                if att_history is None:
                    att_history = alpha_t
                else:
                    att_history = att_history + alpha_t
            else:
                (h_t, cell_t), ctx_t, alpha_t, att_t = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear,
                                                                 None, asp_att_history)
                alpha_asp = torch.zeros_like(alpha_t, dtype=torch.float).scatter_add(1, aspect_idx_var.expand_as(alpha_t), alpha_t)
                alpha1 = torch.clone(alpha_asp)
                alpha1[:, 0] = 0
                alpha_asp = alpha1

                alpha_asp_gather = torch.gather(alpha_asp, 1, aspect_idx_var.expand_as(alpha_asp))

                if asp_att_history is None:
                    asp_att_history = alpha_asp_gather
                else:
                    asp_att_history = asp_att_history + alpha_asp_gather


            # log probabilities over target words
            if self.copy:
		p_gen = F.softmax(self.readout(att_t), dim=-1)
                if max_oov_num > 0:
                    oov_zeros = torch.zeros(p_gen.size(0), max_oov_num, device=self.device)
                    p_gen = torch.cat([p_gen, oov_zeros], dim=-1)

		p = torch.cat([h_t, ctx_t, y_tm1_embed], dim=-1)
		g = torch.sigmoid(self.p_linear(p))

                g = torch.clamp(g, 1e-9, 1 - 1e-9)

		sents_var_complete_src_expanded = src_complete_sents_var.permute(1, 0).expand(alpha_t.size(0), -1)
		p_t = (g * p_gen).scatter_add(1, sents_var_complete_src_expanded, (1 - g) * alpha_t)
		log_p_t = torch.log(p_t)

            else:
                log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)

            assert(len(forbidden_ids)==log_p_t.size(0))
            for forbidden_idx in range(log_p_t.size(0)):
                log_p_t[forbidden_idx][forbidden_ids[forbidden_idx]] = -float('inf')
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            if self.copy:
                prev_hyp_ids = top_cand_hyp_pos / (len(self.vocab.tgt) + max_oov_num)
                hyp_word_ids = top_cand_hyp_pos % (len(self.vocab.tgt) + max_oov_num)
            else:
                prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
                hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            forbidden_ids = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()
                forbidden_id = []

                if self.copy and hyp_word_id >= len(self.vocab.tgt):
                    hyp_word = word_oovs[0][hyp_word_id - len(self.vocab.tgt)]
                else:
                    hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]

                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

                    """ forbid duplicated tri-char """
                    if trichar_repeat_active:
                        if len("".join(new_hyp_sent[1:])) > 2:
                            forbidden_tri_chars = []
                            forbidden_subwords = []

                            for cid in range(len("".join(new_hyp_sent[1:])) - 2):
                                forbidden_tri = "".join(new_hyp_sent[1:])[cid: cid + 3]
                                forbidden_tri_chars.append(forbidden_tri)
                                if forbidden_tri in tri2id:
                                    forbidden_id += tri2id[forbidden_tri]
                                for word_oov in word_oovs[0]:
                                    if forbidden_tri in word_oov:
                                        forbidden_id.append(word_oovs[0].index(word_oov) + len(self.vocab.tgt))

                            for forbidden_tri_char in forbidden_tri_chars:
                                if forbidden_tri_char[0] == "".join(new_hyp_sent[1:])[-1]:
                                    forbidden_bi = forbidden_tri_char[1] + forbidden_tri_char[2]
                                    if forbidden_bi in first_bi2id:
                                        forbidden_id += first_bi2id[forbidden_bi]
                                    for word_oov in word_oovs[0]:
                                        if word_oov and forbidden_bi == word_oov[:2]:
                                            forbidden_id.append(word_oovs[0].index(word_oov) + len(self.vocab.tgt))

                            for forbidden_tri_char in forbidden_tri_chars:
                                if forbidden_tri_char[0] + forbidden_tri_char[1] == "".join(new_hyp_sent[1:])[-2] + \
                                        "".join(new_hyp_sent[1:])[-1]:
                                    forbidden_uni = forbidden_tri_char[2]
                                    if forbidden_uni in first_uni2id:
                                        forbidden_id += first_uni2id[forbidden_uni]
                                    for word_oov in word_oovs[0]:
                                        if word_oov and forbidden_uni == word_oov[0]:
                                            forbidden_id.append(word_oovs[0].index(word_oov) + len(self.vocab.tgt))

                        if aspect_active and aspect_sent and aspect_sent in aspect_map:
                            specify_aspect_map = aspect_map[aspect_sent]

                            new_hyp_sent_segments = re.split('[，？,。?！!]'.decode('utf-8'), " ".join(new_hyp_sent))
                            aspect_key_list = []
                            for segment in new_hyp_sent_segments:
                                for word in segment.strip().split():
                                    word = word.strip()
                                    for key in specify_aspect_map.keys():
                                        if word in specify_aspect_map[key]:
                                            aspect_key_list.append(key)
                                            break
                                    else:
                                        continue
                                    break

                            pre_key = None
                            stop_key_set = set()
                            for key in aspect_key_list:
                                if pre_key and pre_key != key:
                                    stop_key_set.add(pre_key)
                                pre_key = key
                            for key in stop_key_set:
                                for word in specify_aspect_map[key]:
                                    if word in self.vocab.tgt.word2id:
                                        forbidden_id.append(self.vocab.tgt.word2id[word])
                                    else:
                                        if word in word_oovs[0]:
                                            forbidden_id.append(word_oovs[0].index(word) + len(self.vocab.tgt))

                    forbidden_ids.append(forbidden_id)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]
            if self.coverage:
                att_history = att_history[live_hyp_ids]
            asp_att_history = asp_att_history[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @staticmethod
    def reload(argss):
        model_path = argss['--load-model']
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'],
                    copy=str_to_bool(argss['--copy']),
                    coverage=str_to_bool(argss['--coverage']),
                    **args)
        model.load_state_dict(params['state_dict'])

        return model

    @staticmethod
    def load(argss):
        model_path = argss['--model-path']
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'],
                    copy=str_to_bool(argss['--copy']),
                    coverage=str_to_bool(argss['--coverage']),
                    **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

def str_to_bool(str):
    return True if str.lower() == 'true' or str == '1' else False

def evaluate_ppl(model, dev_data, coverage, aspect_map_reversed_idx, batch_size=32):
    """
    Evaluate perplexity on dev sentences

    Args:
        dev_data: a list of dev sentences
        batch_size: batch size

    Returns:
        ppl: the perplexity on dev sentences
    """

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        dev_data_unfold = []
        for src, tgts, img, asp in dev_data:
            for tgt in tgts:
                dev_data_unfold.append([src, tgt, img, asp])

        for src_sents, tgt_sents, imgs, asps in batch_iter(dev_data_unfold, batch_size):
            if coverage:
                loss, coverage_loss = model(src_sents, tgt_sents, imgs, asp)
                loss = -loss.sum()
            else:
                loss, aspect_coverage_loss = model(src_sents, tgt_sents, imgs, asps, aspect_map_reversed_idx)
                loss = -loss.sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    return ppl

def train(args):
    '''
    if str_to_bool(args['--aspect-active']):
        aspect_data_map, _ = read_aspect_file(args['--aspect-file'])
    else:
        aspect_data_map = {}
    '''
    aspect_data_map, aspect_map_reversed, aspect_map_reversed_idx = read_aspect_file(args['--aspect-file'])
    train_txt = read_corpus(args['--train-json'])
    dev_txt = read_corpus(args['--dev-json'])
    img_idx, img_fc = read_image(args['--img'], args['--img-fc'])

    train_data_src = []
    train_data_aspect = []
    train_data_tgt = []
    train_data_img = []
    for sampletuple in train_txt:
        for sku in sampletuple["sku"].split("||"):
            if sku in img_idx:
                src = sampletuple["src"]
                aspect = sampletuple["aspect"]
                img = img_fc[img_idx[sku]]
                for tgt in sampletuple["tgts"]:
                    train_data_src.append(src)
                    train_data_tgt.append(tgt)
                    train_data_img.append(img)
                    train_data_aspect.append(aspect)
                break

    dev_data_src = []
    dev_data_tgt = []
    dev_data_img = []
    dev_data_aspect = []
    for sampletuple in dev_txt:
        dev_data_aspect.append(sampletuple["aspect"])
        for sku in sampletuple["sku"].split("||"):
            if sku in img_idx:
                dev_data_src.append(sampletuple["src"])
                dev_data_tgt.append(sampletuple["tgts"])
                dev_data_img.append(img_fc[img_idx[sku]])
                break

    train_data = list(zip(train_data_src, train_data_tgt, train_data_img, train_data_aspect))
    dev_data = list(zip(dev_data_src, dev_data_tgt, dev_data_img, dev_data_aspect))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    coverage = str_to_bool(args['--coverage'])

    if args['--load-model'] != 'False':
        model_load_path = args['--load-model']
        model_save_path_cur = model_load_path
        print('loading model from %s ...' % model_load_path, file=sys.stderr)
        model = NMT.reload(args)
        hist_valid_rouge_scores = []
        hist_valid_rouge_scores.append(float(args['--best-r2']))
        model.train()
        device = torch.device("cuda:0" if args['--cuda'] else "cpu")
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
        print('restore parameters of the optimizers', file=sys.stderr)
        num_trial = int(args['--num-trial'])

        model.att_c_linear = nn.Linear(1, int(args["--hidden-size"]), bias=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

        lr_cur = float(args['--lr']) * float(args['--lr-decay']) ** num_trial
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_cur

        valid_rouge_score_cur =max(hist_valid_rouge_scores)
        print('current lr is: %f' % lr_cur, file=sys.stderr)
        print('current best rouge2 f1 score is: %f' % float(args['--best-r2']), file=sys.stderr)
    else:
        vocab = Vocab.load(args['--vocab'])
        model = NMT(embed_size=int(args['--embed-size']),
                    hidden_size=int(args['--hidden-size']),
                    dropout_rate=float(args['--dropout']),
                    input_feed=args['--input-feed'],
                    vocab=vocab,
                    copy=str_to_bool(args['--copy']),
                    coverage=coverage)
        num_trial = 0
        hist_valid_rouge_scores = []
        uniform_init = float(args['--uniform-init'])
        if np.abs(uniform_init) > 0.:
            print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
            for p in model.parameters():
                p.data.uniform_(-uniform_init, uniform_init)
        model.train()
        device = torch.device("cuda:0" if args['--cuda'] else "cpu")
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents, img_vecs, batch_aspects in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(src_sents)
            if coverage:
                example_losses, coverage_losses = model(src_sents, tgt_sents, img_vecs, batch_aspects, aspect_map_reversed_idx)
                example_losses = -example_losses
            else:
                example_losses, asp_coverage_losses = model(src_sents, tgt_sents, img_vecs, batch_aspects, aspect_map_reversed_idx)
                example_losses = -example_losses
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            if coverage:
                coverage_loss = coverage_losses.sum() / batch_size
                loss += coverage_loss
            coverage_loss = asp_coverage_losses.sum() / batch_size
            loss += coverage_loss

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                # ROUGE evaluation begin

                dev_hyps = beam_search_dev(model,
                            dev_data_src, dev_data_img, dev_data_aspect, aspect_map_reversed_idx,
                            beam_size=10,
                            max_decoding_time_step=int(args['--max-decoding-time-step']),
                            aspect_map=aspect_data_map, args=args)
                dev_hyps = [hyps[0].value for hyps in dev_hyps]
                dev_rouge2 = get_rouge2f(dev_data_tgt, dev_hyps)

                print('validation: iter %d, dev. ROUGE2 %f' % (train_iter,
                      dev_rouge2), file=sys.stderr)

                model.train()
                is_better = len(hist_valid_rouge_scores) == 0 or dev_rouge2 > max(hist_valid_rouge_scores)
                hist_valid_rouge_scores.append(dev_rouge2)

                if is_better:
                    best_model_iter = train_iter
                    patience = 0
                    model_save_path_cur = model_save_path
                    print('save currently the best model to [%s]' % model_save_path_cur, file=sys.stderr)
                    model.save(model_save_path_cur)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path_cur + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            print('the best model is from iteration [%d]' % best_model_iter,
                                   file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model %s and decay learning rate to %f' % (model_save_path_cur, lr), file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path_cur, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path_cur + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def get_rouge2f(references, hypotheses):
    references = [[[char for char in "".join(ref)] for ref in refs] for refs in references]
    hypotheses = [[char for char in "".join(hyp)] for hyp in hypotheses]
    # compute ROUGE-2 F1-SCORE
    rouge2f_score = rouge_2_corpus_multiple_target(references, hypotheses)
    return rouge2f_score


def beam_search_dev(model, test_data_src, test_data_img, test_data_aspect, aspect_map_reversed_idx, beam_size, max_decoding_time_step, aspect_map, args):
    was_training = model.training
    model.eval()

    first_uni2id = {}
    first_bi2id = {}
    tri2id = {}
    for word, idx in model.vocab.tgt.word2id.items():
        first_uni = word[0]
        if first_uni not in first_uni2id:
            first_uni2id[first_uni] = []
        first_uni2id[first_uni].append(idx)

        if len(word) > 1:
            first_bi = word[:2]
            if first_bi not in first_bi2id:
                first_bi2id[first_bi] = []
            first_bi2id[first_bi].append(idx)

        if len(word) > 2:
            for start_idx in range(len(word) - 2):
                tri = word[start_idx: start_idx + 3]
                if tri not in tri2id:
                    tri2id[tri] = []
                tri2id[tri].append(idx)
    for uni, idx in first_uni2id.items():
        first_uni2id[uni] = list(set(idx))
    for bi, idx in first_bi2id.items():
        first_bi2id[bi] = list(set(idx))
    for tri, idx in tri2id.items():
        tri2id[tri] = list(set(idx))

    hypotheses = []
    begin_time = time.time()
    test_data_tuple = list(zip(test_data_src, test_data_img, test_data_aspect))
    with torch.no_grad():
        for src_sent, img, aspect_sent in tqdm(test_data_tuple, desc='Decoding', file=sys.stdout):
            aspect_idx_dict = aspect_map_reversed_idx[aspect_sent]
            aspect_idx = [aspect_idx_dict.get(w, 0) for w in src_sent]
            example_hyps = model.beam_search(src_sent, img, aspect_sent, aspect_idx, first_uni2id, first_bi2id, tri2id, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, aspect_map=aspect_map, args=args)

            hypotheses.append(example_hyps)

    elapsed = time.time() - begin_time
    print('decoded %d examples, took %d s' % (len(test_data_src), elapsed), file=sys.stderr)

    if was_training:
        model.train(was_training)

    return hypotheses

def decode(args):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    aspect_data_map, aspect_map_reversed, aspect_map_reversed_idx = read_aspect_file(args['--aspect-file'])

    print("load test source sentences from %s" % args['--test-json'], file=sys.stderr)
    test_txt = read_corpus(args['--test-json'])
    img_idx, img_fc = read_image(args['--img'], args['--img-fc'])

    test_data_src = []
    test_data_tgt = []
    test_data_img = []
    test_data_aspect = []
    test_sku = []
    for sampletuple in test_txt:
        test_data_aspect.append(sampletuple["aspect"])
        for sku in sampletuple["sku"].split("||"):
            if sku in img_idx:
                test_data_src.append(sampletuple["src"])
                test_data_tgt.append(sampletuple["tgts"])
                test_data_img.append(img_fc[img_idx[sku]])
                test_sku.append(sampletuple["sku"])
                break

    print("load model from %s " % args['--model-path'], file=sys.stderr)
    model = NMT.load(args)

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search_dev(model, test_data_src, test_data_img,
                             test_data_aspect, aspect_map_reversed_idx,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             aspect_map=aspect_data_map,
                             args=args)

    with open(args['--output'], 'w') as f:
        for sku, hyps in zip(test_sku, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent.encode('utf-8') + '\n')

def main():
    args = docopt(__doc__)
    print('Current Time: '+ str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

    print(args, file=sys.stderr)
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['--mode'] == 'train':
        train(args)
    elif args['--mode'] == 'decode':
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
