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
"""seq2seq的Pytorch实现核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """ seq2seq的encoder """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, num_layers: int,
                 dropout: float, cell_type: str = "lstm", if_bidirectional: bool = True) -> None:
        """
        :param vocab_size: 词汇量大小
        :param embedding_dim: 词嵌入维度
        :param enc_units: encoder单元大小
        :param num_layers: encoder中内部RNN层数
        :param dropout: 采样率
        :param if_bidirectional: 是否双向
        :param cell_type: cell类型，lstm/gru， 默认lstm
        :return: Seq2Seq的Encoder
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = list()

        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=enc_units,
                               num_layers=num_layers, bidirectional=if_bidirectional)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=enc_units,
                              num_layers=num_layers, bidirectional=if_bidirectional)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: encoder的输入
        """
        inputs = self.embedding(inputs)
        dropout = self.dropout(inputs)
        outputs, state = self.rnn(dropout)
        # 这里使用了双向GRU，所以这里将两个方向的特征层合并起来，维度将会是units * 2
        state = torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)
        return outputs, state


class Decoder(nn.Module):
    """ seq2seq的decoder

    :param vocab_size: 词汇量大小
    :param embedding_dim: 词嵌入维度
    :param enc_units: encoder单元大小
    :param dec_units: decoder单元大小
    :param num_layers: encoder中内部RNN层数
    :param dropout: 采样率
    :param attention: 用于计算attention
    :param cell_type: cell类型，lstm/gru， 默认lstm
    :param if_bidirectional: 是否双向
    :return: Seq2Seq的Encoder
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, dec_units: int, num_layers: int,
                 dropout: float, attention: nn.Module, cell_type: str = "lstm", if_bidirectional: bool = True) -> None:
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size=enc_units * 2 + embedding_dim, hidden_size=dec_units,
                               num_layers=num_layers, bidirectional=if_bidirectional)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size=enc_units * 2 + embedding_dim, hidden_size=dec_units,
                              num_layers=num_layers, bidirectional=if_bidirectional)
        self.fc = nn.Linear(in_features=2 * enc_units + 2 * dec_units + embedding_dim, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor,
                enc_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: decoder的输入
        :param hidden: encoder的hidden
        :param enc_output: encoder的输出
        """
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        rnn_input = torch.cat((embedding, torch.unsqueeze(context_vector, dim=0)), dim=-1)
        rnn_output, dec_state = self.rnn(rnn_input, hidden.unsqueeze(dim=0))
        embedding = embedding.squeeze(dim=0)
        rnn_output = rnn_output.squeeze(dim=0)
        output = self.fc(torch.cat((embedding, context_vector, rnn_output), dim=-1))

        return output, dec_state.squeeze(0)
