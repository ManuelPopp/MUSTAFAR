#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Manuel"
__date__ = "Thu Dec 05 14:45:21 2024"
__credits__ = ["Manuel R. Popp"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
# Imports
import math
import torch
import torch.nn as nn

#-----------------------------------------------------------------------------|
# Classes
class DisturbanceTransformer(nn.Module):
    def __init__(
        self, n_features, d_model, n_heads, n_layers, n_classes, sequence_length
        ):
        super(DisturbanceTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead = n_heads, batch_first = True
            )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers = n_layers
            )
        self.fc_out = nn.Linear(d_model, n_classes)
    
    def forward(self, x, attention_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask = attention_mask)
        logits = self.fc_out(x)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
