#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:09:55 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from Utils import Utils

class EncoderLayer(nn.Module):
    
    def __init__(self):
        
        self.d_model = 512
        self.d_ff = 2048
        
        super(EncoderLayer, self).__init__()
        self.utils = Utils()
        
        self.new_representation_affine_transformation = nn.Sequential([nn.Linear(self.d_model, self.d_ff),
                                                                  nn.ReLU(),
                                                                  nn.Linear(self.d_ff, self.d_model)])
        
    def forward(self, X):
        
        self.multi_head_attention = MultiHeadAttention(8, X, X, X)
        intermediate_representation = self.multi_head_attention()
        
        layer_norm1 = nn.LayerNorm(intermediate_representation.shape)
        intermediate_representation = layer_norm1(intermediate_representation + X)
            
        new_representation = self.new_representation_affine_transformation(intermediate_representation)
        
        layer_norm2 = nn.LayerNorm(new_representation.shape)
        new_representation = layer_norm2(new_representation + intermediate_representation)
        
        return new_representation