#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:42:20 2019

@author: assiene
"""

import nltk

class Lang:
    
    def __init__(self):
        self.SOS_token = 0
        self.EOS_token = 1
        self.word2index = {"<SOS>": self.SOS_token, "<EOS>": self.EOS_token}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2
    
    def update_word2index(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            self.word2index[token] = self.n_words
            self.index2word[self.n_words] = token
            self.n_words+= 1
            
    def sentence2tokens(self, sentence):
        output_tokens = []
        words = nltk.word_tokenize(sentence)
        for word in words:
            output_tokens.append(self.word2index[word])
            
        output_tokens.append(self.EOS_token)
        
        return output_tokens
    
    def tokens2sentence(self, tokens):
        sentence = []
        for token in tokens:
            sentence.append(self.index2word[token])
            
        return sentence