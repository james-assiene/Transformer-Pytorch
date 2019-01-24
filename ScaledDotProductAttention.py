#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:38:39 2018

@author: assiene
"""

import numpy as np
import torch

def scaled_dot_product_attention(Q, K, V, mask=False):
    d_k = K.size(2) # because 0 is the batch dimension, 1 the individual dimension
    attention_scores = Q @ K.t() / torch.sqrt(d_k)

    if mask:
        attention_scores = torch.tril(attention_scores)
        attention_scores[attention_scores == 0] = -float("Inf")
    
    attention_weights = attention_scores.softmax(1)
    
    new_representation = attention_weights @ V
    
    return new_representation