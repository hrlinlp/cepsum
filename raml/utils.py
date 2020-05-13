# coding=utf-8
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import re
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json

def read_image(img_file_path, img_vec_file_path):
    image_name_idx = {}
    idx = 0
    for line in open(img_file_path).readlines():
        image_name_idx[line.strip().split(".")[0]] = idx
        idx += 1
    image_feature = np.load(img_vec_file_path)
    return image_name_idx, image_feature

def read_test_corpus(file_path):
    input_lines = open(file_path).readlines()
    data_src = []
    data_aspect = []
    for line in input_lines:
        segments = line.strip().split('\t')
        if len(segments) == 2:  # source + '\t' + aspect
            src = [word.decode("utf-8") for word in segments[0].strip().split(' ')]
            aspect = segments[1].strip()
            data_src.append(src)
            data_aspect.append(aspect)
        else:
            src = [word.decode("utf-8") for word in segments[0].strip().split(' ')]
            data_src.append(src)
            data_aspect.append("")

    return data_src, data_aspect

""" 读入所有品类的aspect词表 """
def read_aspect_file(file_path):
    filemaps = open(file_path + "/maps")
    maps = {}
    for line in filemaps:
        k = line.strip().split(":")[0].strip().decode("utf-8")
        vs = line.strip().split(":")[1].strip().split()
        for v in vs:
            maps[v.decode("utf-8")] = k
    result_map = {}
    files = os.listdir(file_path)
    for file_name in files:
        if file_name != "maps":
            aspect_map = json.load(open(file_path + file_name))
            result_map[unicode(file_name, "utf-8")] = {}
            for item in aspect_map['aspects']:
                result_map[unicode(file_name, "utf-8")][item['name']] = set(item['keywords'])

    aspect_map = {}
    for k, v in result_map.items():
        aspect_map[k] = v
    for k, v in maps.items():
        aspect_map[k] = result_map[v]

    aspect_map_reversed = {}
    for cate, info in aspect_map.items():
        aspect_map_reversed[cate] = {}
        for k, vs in info.items():
            for v in vs:
                aspect_map_reversed[cate][v] = k

    return aspect_map, aspect_map_reversed


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path):
    data = []
    fjson = json.load(open(file_path))
    for sku in fjson.keys():
        sampletuple = {}
        sampletuple["sku"] = sku
        sampletuple["src"] = [w.decode("utf-8") for w in fjson[sku]["src"].split()]
        sampletuple["tgts"] = []
        sampletuple["aspect"] = fjson[sku]["cate"]
        for target_text in fjson[sku]["tgt"]:
            sampletuple["tgts"].append(['<s>'] + [w.decode("utf-8") for w in target_text.split()] + ['</s>'])
        data.append(sampletuple)

    return data

def read_corpus_json(file_path):
    res = json.load(open(file_path))
    data_src = []
    data_tgt = []
    data_aspect = []  # 增加 aspect list
    for sku, txt in res.items():
        src = txt["src"]
        tgts = txt["tgt"]
        aspect = txt["cate"]
        src = [word for word in src.strip().split(' ')]
        tgts = [[word for word in tgt.strip().split(' ')] for tgt in tgts]
        data_src.append(src)
        data_tgt.append(tgts)
        data_aspect.append(aspect)
    return data_src, data_tgt, data_aspect


def batch_iter(data, batch_size, shuffle=False):
    batch_num = int(math.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        img_vecs = [e[2] for e in examples]
        aspects = [e[3] for e in examples]

        yield src_sents, tgt_sents, img_vecs, aspects


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target, max_oov_num=0, copy=False):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        if copy:
            oov_zeros = torch.zeros(true_dist.size(0), max_oov_num, device=torch.device("cuda:0"))
            true_dist = torch.cat([true_dist, oov_zeros], dim=-1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)


        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss
