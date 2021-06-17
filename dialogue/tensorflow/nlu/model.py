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
"""transformer模型核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dialogue.tensorflow.layers import MultiHeadAttention
from dialogue.tensorflow.utils import combine_mask
from dialogue.tensorflow.utils import create_padding_mask
from dialogue.tensorflow.positional_encoding import positional_encoding


def encoder(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """transformer的encoder

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Encoder
    """
    inputs = tf.keras.Input(shape=(None,), name="{}_inputs".format(name), dtype=d_type)
    padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                          name="{}_padding_mask".format(name))(inputs)
    embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                           dtype=d_type, name="{}_embeddings".format(name))(inputs)
    embeddings *= tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(units=units, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                d_type=d_type, name="{}_layer_{}".format(name, i))([outputs, padding_mask])

    return tf.keras.Model(inputs=inputs, outputs=[outputs, padding_mask], name=name)


def decoder(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "decoder") -> tf.keras.Model:
    """transformer的decoder

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder
    """
    inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(None, embedding_dim), dtype=d_type, name="{}_encoder_outputs".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    look_ahead_mask = tf.keras.layers.Lambda(combine_mask, output_shape=(1, None, None),
                                             name="{}_look_ahead_mask".format(name))(inputs)
    embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, dtype=d_type,
                                           name="{}_embeddings".format(name))(inputs)

    embeddings *= tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units, d_model=embedding_dim, num_heads=num_heads, dropout=dropout, d_type=d_type,
            name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, dtype=d_type, name="{}_outputs_dense".format(name))(outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs, padding_mask], outputs=outputs, name=name)


def encoder_layer(units: int, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "encoder_layer") -> tf.keras.Model:
    """Transformer的encoder层

    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Encoder内部层
    """
    inputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_inputs".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention, _ = MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs, mask=padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_attention_dropout".format(name))(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                   name="{}_attention_layer_norm".format(name))(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation="relu",
                                    dtype=d_type, name="{}_dense_act".format(name))(attention)
    outputs = tf.keras.layers.Dense(units=d_model, dtype=d_type, name="{}_dense".format(name))(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units: int, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "decoder_layer") -> tf.keras.Model:
    """Transformer的decoder层

    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder内部层
    """
    inputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_encoder_outputs".format(name))
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), dtype=d_type, name="{}_look_ahead_mask".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention1, _ = MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs, mask=look_ahead_mask)
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm1".format(name))(attention1 + inputs)

    attention2, _ = MultiHeadAttention(d_model, num_heads)(
        q=attention1, k=enc_outputs, v=enc_outputs, mask=padding_mask)
    attention2 = tf.keras.layers.Dropout(
        rate=dropout, dtype=d_type, name="{}_attention_drouput".format(name))(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm2".format(name))(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation="relu",
                                    dtype=d_type, name="{}_dense_act".format(name))(attention2)
    outputs = tf.keras.layers.Dense(units=d_model, dtype=d_type, name="{}_dense".format(name))(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

