#!/usr/bin/env python
"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import json
import torch

from utils import read_corpus, input_transpose


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3

        self.unk_id = self.word2id['<unk>']

        self.id2word = {v: k for k, v in self.word2id.items()}


    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            # This way
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def words2indices_complete_decode(self, sents_src):
        word_src_complete_ids = []
        word_oovs = []
        vocab_size = len(self.word2id)
        max_oov_num = 0
        for sent_src in sents_src:
            word_oov_s = []
            word_src_complete_id_s = []
            for w in sent_src:
                if w not in self.word2id:
                    if w not in word_oov_s:
                        word_oov_s.append(w)
                    word_src_complete_id_s.append(vocab_size + word_oov_s.index(w))
                else:
                    word_src_complete_id_s.append(self.word2id[w])
            word_src_complete_ids.append(word_src_complete_id_s)
            word_oovs.append(word_oov_s)
            if max_oov_num < len(word_oov_s):
                max_oov_num = len(word_oov_s)
        return word_src_complete_ids, word_oovs, max_oov_num

    def words2indices_complete(self, sents_src, sents_tgt):
        assert(len(sents_src) == len(sents_tgt))
        word_tgt_ids = [[self[w] for w in s] for s in sents_tgt]
        word_src_complete_ids = []
        word_tgt_complete_ids = []
        word_oovs = []
        vocab_size = len(self.word2id)
        max_oov_num = 0
        for sent_src, sent_tgt in zip(sents_src, sents_tgt):
            word_oov_s = []
            word_src_complete_id_s = []
            word_tgt_complete_id_s = []
            for w in sent_src:
                if w not in self.word2id:
                    if w not in word_oov_s:
                        word_oov_s.append(w)
                    word_src_complete_id_s.append(vocab_size + word_oov_s.index(w))
                else:
                    word_src_complete_id_s.append(self.word2id[w])
            word_src_complete_ids.append(word_src_complete_id_s)
            word_oovs.append(word_oov_s)
            if max_oov_num < len(word_oov_s):
                max_oov_num = len(word_oov_s)
            for w in sent_tgt:
                if w not in self.word2id:
                    if w in word_oov_s:
                        word_tgt_complete_id_s.append(vocab_size + word_oov_s.index(w))
                    else:
                        word_tgt_complete_id_s.append(self.unk_id)
                else:
                    word_tgt_complete_id_s.append(self.word2id[w])
            word_tgt_complete_ids.append(word_tgt_complete_id_s)
        return word_tgt_ids, word_src_complete_ids, word_tgt_complete_ids, word_oovs, max_oov_num

    def to_input_tensor_src(self, sents_src, device):
        word_src_ids = self.words2indices(sents_src)

        sents_t_src = input_transpose(word_src_ids, self['<pad>'])
        sents_var_src = torch.tensor(sents_t_src, dtype=torch.long, device=device)

        return sents_var_src

    def to_input_tensor_tgt_decode(self, sents_src, device):
        (word_src_complete_ids, word_oovs, max_oov_num) = self.words2indices_complete_decode(sents_src)
        sents_t_src_complete = input_transpose(word_src_complete_ids, self['<pad>'])
        sents_var_complete_src = torch.tensor(sents_t_src_complete, dtype=torch.long, device=device)

        return sents_var_complete_src, word_oovs, max_oov_num

    def to_input_tensor_tgt(self, sents_src, sents_tgt, device):
        (word_tgt_ids, word_src_complete_ids, word_tgt_complete_ids,
                word_oovs, max_oov_num) = self.words2indices_complete(sents_src, sents_tgt)
        sents_t_src_complete = input_transpose(word_src_complete_ids, self['<pad>'])
        sents_t_tgt_complete = input_transpose(word_tgt_complete_ids, self['<pad>'])
        sents_var_complete_src = torch.tensor(sents_t_src_complete, dtype=torch.long, device=device)
        sents_var_complete_tgt = torch.tensor(sents_t_tgt_complete, dtype=torch.long, device=device)
        sents_t_tgt = input_transpose(word_tgt_ids, self['<pad>'])
        sents_var_tgt = torch.tensor(sents_t_tgt, dtype=torch.long, device=device)

        return sents_var_tgt, sents_var_complete_src, sents_var_complete_tgt, word_oovs, max_oov_num

    @staticmethod
    def from_corpus(words):
        vocab_entry = VocabEntry()

        for word in words:
            vocab_entry.add(word.encode('utf-8'))

        return vocab_entry

class Vocab(object):
    def __init__(self, src_vocab, tgt_vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents):

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

if __name__ == '__main__':

    args = docopt(__doc__)

    print('read in chars: %s' % args['--word'])

    fin_src = open(args['--train-src']).readlines()
    fin_tgt = open(args['--train-tgt']).readlines()
    words_src = []
    words_tgt = []
    for line in fin_src:
        words_src.append(line.strip().split()[0].decode('utf-8'))
    for line in fin_tgt:
        words_tgt.append(line.strip().split()[0].decode('utf-8'))

    vocab = Vocab.build(words_src, words_tgt)
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
