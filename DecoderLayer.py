#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:56:24 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from Utils import Utils

class DecoderLayer(nn.Module):
    
    def __init__(self, output_dictinonary_size):
        
        self.d_model = 512
        self.d_ff = 2048
        self.output_dictionnary_size = output_dictinonary_size
        
        super(DecoderLayer, self).__init__()
        self.utils = Utils()
        
        self.new_representation_affine_transformation = nn.Sequential([nn.Linear(self.d_model, self.d_ff),
                                                                  nn.ReLU(),
                                                                  nn.Linear(self.d_ff, self.d_model)])
    
        self.output_affine_transformation = nn.Linear(self.d_model, self.output_dictionnary_size)
        
        
    def forward(self, output_embedding, encoder_output):
        
        self.masked_multi_head_attention = MultiHeadAttention(8, output_embedding, output_embedding, output_embedding)
        output_intermediate_representation = self.masked_multi_head_attention(mask=True)
        
        layer_norm0 = nn.LayerNorm(output_intermediate_representation.shape)
        output_intermediate_representation = layer_norm0(output_intermediate_representation + output_embedding)
                
        self.multi_head_attention = MultiHeadAttention(8, output_intermediate_representation, encoder_output, encoder_output)
        intermediate_representation = self.multi_head_attention()
        
        layer_norm1 = nn.LayerNorm(intermediate_representation.shape)
        intermediate_representation = layer_norm1(intermediate_representation + output_intermediate_representation)
        
        new_representation = self.new_representation_affine_transformation(intermediate_representation)
        
        layer_norm2 = nn.LayerNorm(new_representation.shape)
        new_representation = layer_norm2(new_representation + intermediate_representation)
        
        output = self.output_affine_transformation(new_representation[:,-1,:])
        
        output = nn.functional.log_softmax(output, dim=1)
        
        return output, new_representation