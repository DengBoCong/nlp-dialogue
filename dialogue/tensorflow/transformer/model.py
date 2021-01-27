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
                                          name="{}_padding_mask".format(name))(inputs, d_type)
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
                                             name="{}_look_ahead_mask".format(name))(inputs, d_type)
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


def gumbel_softmax(inputs: tf.Tensor, alpha: float):
    """
    按照论文中的公式，实现GumbelSoftmax，具体见论文公式
    :param inputs: 输入
    :param alpha: 温度
    :return: 混合Gumbel噪音后，做softmax以及argmax之后的输出
    """
    uniform = tf.random.uniform(shape=tf.shape(inputs), maxval=1, minval=0)
    # 以给定输入的形状采样Gumbel噪声
    gumbel_noise = -tf.math.log(-tf.math.log(uniform))
    # 将Gumbel噪声添加到输入中，输入第三维就是分数
    gumbel_outputs = inputs + gumbel_noise
    gumbel_outputs = tf.cast(gumbel_outputs, dtype=tf.float32)
    # 在给定温度下，进行softmax并返回
    gumbel_outputs = tf.nn.softmax(alpha * gumbel_outputs)
    gumbel_outputs = tf.argmax(gumbel_outputs, axis=-1)
    return tf.cast(gumbel_outputs, dtype=tf.float32)


def embedding_mix(gumbel_inputs: tf.Tensor, inputs: tf.Tensor):
    """
    将输入和gumbel噪音混合嵌入，线性衰减
    :param gumbel_inputs: 噪音输入
    :param inputs: 输入
    :return: 混合嵌入
    """
    probability = tf.random.uniform(shape=tf.shape(inputs), maxval=1, minval=0, dtype=tf.float32)
    return tf.where(probability < 0.3, x=gumbel_inputs, y=inputs)


def transformer_scheduled_sample(vocab_size, num_layers, units, d_model, num_heads,
                                 dropout, alpha=1.0, name="transformer_scheduled_sample") -> tf.keras.Model:
    """
    Transformer应用Scheduled Sample
    :param vocab_size: token大小
    :param num_layers: 编码解码层的数量
    :param units: 单元大小
    :param d_model: 词嵌入维度
    :param num_heads:多头注意力的头部层数量
    :param dropout: dropout的权重
    :param alpha: 温度
    :param name: 名称
    :return: Scheduled Sample的Transformer
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 使用了Lambda将方法包装成层，为的是满足函数式API的需要
    enc_padding_mask = tf.keras.layers.Lambda(
        layers.create_padding_mask, output_shape=(1, 1, None),
        name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        _combine_mask, output_shape=(1, None, None),
        name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        layers.create_padding_mask, output_shape=(1, 1, None),
        name="dec_padding_mask"
    )(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )(inputs=[inputs, enc_padding_mask])

    transformer_decoder = decoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        d_model=d_model, num_heads=num_heads, dropout=dropout
    )

    dec_first_outputs = transformer_decoder(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # dec_outputs的几种方式
    # 1. dec_outputs = tf.argmax(dec_outputs, axis=-1)  # 使用这个方式的话，就是直接返回最大的概率用来作为decoder的inputs
    # 2. tf.layers.Sparsemax(axis=-1)(dec_outputs) # 使用Sparsemax的方法，具体公式参考论文
    # 3. tf.math.top_k() # 混合top-k嵌入，使用得分最高的5个词汇词嵌入的加权平均值。
    # 4. 使用GumbelSoftmax的方法，具体公式参考论文，下面就用GumbelSoftmax方法
    # 这里使用论文的第四种方法：GumbelSoftmax
    gumbel_outputs = gumbel_softmax(dec_first_outputs, alpha=alpha)
    dec_first_outputs = embedding_mix(gumbel_outputs, dec_inputs)

    dec_second_outputs = transformer_decoder(inputs=[dec_first_outputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_second_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def accuracy(real, pred):
    real = tf.reshape(real, shape=(-1, 40 - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(real, pred)


def _combine_mask(seq: tf.Tensor):
    """
    对input中的不能见单位进行mask
    :param seq: 输入序列
    :return: mask
    """
    look_ahead_mask = layers.create_look_ahead_mask(seq)
    padding_mask = layers.create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)
