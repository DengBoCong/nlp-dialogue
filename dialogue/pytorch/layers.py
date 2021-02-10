# Copyright 2021 DengBoCong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""公用层组件
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import NoReturn


class BahdanauAttention(nn.Module):
    """ bahdanau attention实现

    :param enc_units: encoder单元大小
    :param dec_units: decoder单元大小
    """

    def __init__(self, enc_units: int, dec_units: int) -> NoReturn:
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=2 * enc_units, out_features=dec_units)
        self.W2 = nn.Linear(in_features=2 * enc_units, out_features=dec_units)
        self.V = nn.Linear(in_features=dec_units, out_features=1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: 隐层状态
        :param values: encoder输出状态
        """
        values = values.permute(1, 0, 2)
        hidden_with_time_axis = torch.unsqueeze(input=query, dim=1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = F.softmax(input=score, dim=1)
        context_vector = attention_weights * values
        context_vector = torch.sum(input=context_vector, dim=1)

        return context_vector, attention_weights


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """ PE(pos,2i) = sin(pos/10000^(2i/d_model)) | PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

        :param d_model: 词嵌入大小
        :param dropout: 采样率
        :param max_len: 最大位置长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
