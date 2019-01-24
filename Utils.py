#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:16:07 2018

@author: assiene
"""

import numpy as np
import torch

class Utils:
    def __init__(self):
        pass
    
    def get_positional_enconding(self, words):
        d_model = words.size(2)
        dim_index = torch.arange(0, d_model / 2)
        positions = torch.arange(0, words.size(1))
        PE = torch.empty(words.size(1), words.size(2))
        
        for i in dim_index:
            if i % 2 == 0:
                PE[:,i] = torch.sin(positions/torch.pow(10000, 2 * i / d_model))
            else:
                PE[:,i] = torch.cos(positions/torch.pow(10000, 2 * i / d_model))
                
        return PE