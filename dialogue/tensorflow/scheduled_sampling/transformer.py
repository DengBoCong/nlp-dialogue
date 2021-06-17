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
"""应用scheduled_sampling的transformer模型核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dialogue.tensorflow.nlu.model import decoder
from dialogue.tensorflow.nlu.model import encoder
from dialogue.tensorflow.utils import combine_mask
from dialogue.tensorflow.utils import create_padding_mask


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
        create_padding_mask, output_shape=(1, 1, None),
        name="enc_padding_mask"
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        combine_mask, output_shape=(1, None, None),
        name="look_ahead_mask"
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
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
