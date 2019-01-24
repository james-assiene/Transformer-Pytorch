#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 19:14:10 2018

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn

from ScaledDotProductAttention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, h, Q, K, V):
        """
        Constructor
        
        :param h : Number of parallel attention layers
        :param d_model : dimension of the queries, keys and values (words) representations
        :parma d_v : dimension 
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.Q = Q
        self.K = K
        self.V = V
        
        self.d_model = Q.size(2)
        self.d_v = self.d_model / self.h
        self.d_k = self.d_model / self.h
        
        self.queries_projections = [nn.Linear(self.d_model, self.d_k, bias = False) for i in range(h)]
        self.keys_projections = [nn.Linear(self.d_model, self.d_k, bias = False) for i in range(h)]
        self.values_projections = [nn.Linear(self.d_model, self.d_v, bias = False) for i in range(h)]
        
        self.output_projection = nn.Linear(self.d_v * self.h, self.d_model)
        
        
    def forward(self, mask=False):
        new_values_representations = []
        for i in range(self.h):
            projected_queries = self.queries_projections[i](self.Q)
            projected_keys = self.keys_projections[i](self.K)
            projected_values = self.values_projections[i](self.V)
            
            new_values_representation = scaled_dot_product_attention(projected_queries, projected_keys, projected_values, mask)
            new_values_representations.append(new_values_representation)
            
        new_values_representation = torch.cat(new_values_representations, dim = 1) #1 because 0 is the batch dimension
        
        return self.output_projection(new_values_representation)  
        
        
        
        