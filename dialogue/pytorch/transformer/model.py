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
"""transformer的Pytorch实现核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from typing import Any, Optional, NoReturn


class Transformer(torch.nn.Module):
    """ Transformer Model """

    def __init__(self, d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 units: int = 2048, dropout: float = 0.1, activation: str = "relu") -> NoReturn:
        """
        :param d_model: 深度，词嵌入维度
        :param num_heads: 注意力头数
        :param num_encoder_layers: encoder层数
        :param num_decoder_layers: decoder层数
        :param units: 单元数
        :param dropout: 采样率
        :param activation: 激活方法
        """
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, num_heads, units, dropout, activation)
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, num_heads, units, dropout, activation)
        decoder_norm = torch.nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor, enc_mask: Optional[torch.Tensor] = None,
                dec_mask: Optional[torch.Tensor] = None, enc_outputs_mask: Optional[torch.Tensor] = None,
                enc_key_padding_mask: Optional[torch.Tensor] = None, dec_key_padding_mask: Optional[torch.Tensor] = None,
                enc_outputs_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param enc_inputs: encoder 输入
        :param dec_inputs: decoder 输入
        :param enc_mask: encoder 输入序列的mask
        :param dec_mask: decoder 输入序列的mask
        :param enc_outputs_mask: encoder 输出序列的mask
        :param enc_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
        :param dec_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
        :param enc_outputs_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        """

        if enc_inputs.size(1) != dec_inputs.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if enc_inputs.size(2) != self.d_model or dec_inputs.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(enc_inputs, mask=enc_mask, src_key_padding_mask=enc_key_padding_mask)
        output = self.decoder(dec_inputs, memory, tgt_mask=dec_mask, memory_mask=enc_outputs_mask,
                              tgt_key_padding_mask=dec_key_padding_mask,
                              memory_key_padding_mask=enc_outputs_key_padding_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
