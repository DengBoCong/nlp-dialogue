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
"""smn检索式模型实现核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def accumulate(units: int, embedding_dim: int, max_utterance: int, max_sentence: int,
               d_type: tf.dtypes.DType = tf.float32, name: str = "accumulate") -> tf.keras.Model:
    """ SMN的语义抽取层，主要是对匹配对的两个相似度矩阵进行计算，并返回最终的最后一层GRU的状态，用于计算分数

    :param units: GRU单元数
    :param embedding_dim: embedding维度
    :param max_utterance: 每轮最大语句数
    :param max_sentence: 句子最大长度
    :param d_type: 运算精度
    :param name: 名称
    :return: GRU的状态
    """
    utterance_inputs = tf.keras.Input(shape=(max_utterance, max_sentence, embedding_dim),
                                      dtype=d_type, name="{}_utterance_inputs".format(name))
    response_inputs = tf.keras.Input(shape=(max_sentence, embedding_dim),
                                     dtype=d_type, name="{}_response_inputs".format(name))
    a_matrix = tf.keras.initializers.GlorotNormal()(shape=(units, units), dtype=d_type)

    # 这里对response进行GRU的Word级关系建模，这里用正交矩阵初始化内核权重矩阵，用于输入的线性变换。
    response_gru = tf.keras.layers.GRU(units=units, return_sequences=True, kernel_initializer="orthogonal",
                                       dtype=d_type, name="{}_gru".format(name))(response_inputs)
    conv2d_layer = tf.keras.layers.Conv2D(
        filters=8, kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal",
        activation="relu", dtype=d_type, name="{}_conv2d".format(name)
    )
    max_pooling2d_layer = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(3, 3), padding="valid", dtype=d_type, name="{}_pooling2d".format(name)
    )
    dense_layer = tf.keras.layers.Dense(
        50, activation="tanh", kernel_initializer="glorot_normal", dtype=d_type, name="{}_dense".format(name)
    )

    # 这里需要做一些前提工作，因为我们要针对每个batch中的每个utterance进行运算，所
    # 以我们需要将batch中的utterance序列进行拆分，使得batch中的序列顺序一一匹配
    utterance_embeddings = tf.unstack(utterance_inputs, num=max_utterance, axis=1, name="{}_unstack".format(name))
    matching_vectors = []
    for index, utterance_input in enumerate(utterance_embeddings):
        # 求解第一个相似度矩阵，公式见论文
        matrix1 = tf.matmul(utterance_input, response_inputs, transpose_b=True, name="{}_matmul_{}".format(name, index))
        utterance_gru = tf.keras.layers.GRU(units, return_sequences=True, kernel_initializer="orthogonal",
                                            dtype=d_type, name="{}_gru_{}".format(name, index))(utterance_input)
        matrix2 = tf.einsum("aij,jk->aik", utterance_gru, a_matrix)
        # matrix2 = tf.matmul(utterance_gru, a_matrix)
        # 求解第二个相似度矩阵
        matrix2 = tf.matmul(matrix2, response_gru, transpose_b=True)
        matrix = tf.stack([matrix1, matrix2], axis=3)

        conv_outputs = conv2d_layer(matrix)
        pooling_outputs = max_pooling2d_layer(conv_outputs)
        flatten_outputs = tf.keras.layers.Flatten(dtype=d_type, name="{}_flatten_{}".format(name, index))(
            pooling_outputs)

        matching_vector = dense_layer(flatten_outputs)
        matching_vectors.append(matching_vector)

    vector = tf.stack(matching_vectors, axis=1, name="{}_stack".format(name))
    outputs = tf.keras.layers.GRU(
        units, kernel_initializer="orthogonal", dtype=d_type, name="{}_gru_outputs".format(name)
    )(vector)

    return tf.keras.Model(inputs=[utterance_inputs, response_inputs], outputs=outputs)


def smn(units: int, vocab_size: int, embedding_dim: int, max_utterance: int, max_sentence: int,
        d_type: tf.dtypes.DType = tf.float32, name: str = "smn") -> tf.keras.Model:
    """ SMN的模型，在这里将输入进行accumulate之后，得到匹配对的向量，然后通过这些向量计算最终的分类概率

    :param units: GRU单元数
    :param vocab_size: embedding词汇量
    :param embedding_dim: embedding维度
    :param max_utterance: 每轮最大语句数
    :param max_sentence: 句子最大长度
    :param d_type: 运算精度
    :param name: 名称
    :return: 匹配对打分
    """
    utterances = tf.keras.Input(shape=(max_utterance, max_sentence), dtype=d_type, name="{}_utterance".format(name))
    responses = tf.keras.Input(shape=(max_sentence,), dtype=d_type, name="{}_response".format(name))

    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, dtype=d_type, name="{}_embedding".format(name))
    utterances_embeddings = embeddings(utterances)
    responses_embeddings = embeddings(responses)

    accumulate_outputs = accumulate(
        units=units, embedding_dim=embedding_dim, max_utterance=max_utterance,
        max_sentence=max_sentence, d_type=d_type, name="{}_accumulate".format(name)
    )(inputs=[utterances_embeddings, responses_embeddings])

    outputs = tf.keras.layers.Dense(
        2, kernel_initializer="glorot_normal", dtype=d_type, name="{}_dense_outputs".format(name)
    )(accumulate_outputs)

    outputs = tf.keras.layers.Softmax(axis=-1, dtype=d_type, name="{}_softmax".format(name))(outputs)

    return tf.keras.Model(inputs=[utterances, responses], outputs=outputs)
