#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:41:33 2019

@author: assiene
"""

import torch 
import torchtext
import numpy as np
from torchtext import data

src = data.Field(tokenize="spacy")
trg = data.Field(tokenize="spacy")

mt_train = torchtext.datasets.TranslationDataset(path="./data/europarl-v7.fr-en", exts=(".en", ".fr"), fields=(src, trg))

src.build_vocab(mt_train, max_size=80000)
trg.build_vocab(mt_train, max_size=40000)

train_iter = data.BucketIterator(dataset=mt_train, batch_size=32, sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))