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
"""seq2seq模型核心core
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dialogue.tensorflow.layers import bahdanau_attention
from dialogue.tools import log_operator


def rnn_layer(units: int, input_feature_dim: int, cell_type: str = 'lstm', if_bidirectional: bool = True,
              d_type: tf.dtypes.DType = tf.float32, name: str = "rnn_layer") -> tf.keras.Model:
    """ RNNCell层，其中可定义cell类型，是否双向

    :param units: cell单元数
    :param input_feature_dim: 输入的特征维大小
    :param cell_type: cell类型，lstm/gru， 默认lstm
    :param if_bidirectional: 是否双向
    :param d_type: 运算精度
    :param name: 名称
    :return: Multi-layer RNN
    """
    inputs = tf.keras.Input(shape=(None, input_feature_dim), dtype=d_type, name="{}_inputs".format(name))
    if cell_type == 'lstm':
        rnn = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True,
                                   recurrent_initializer='glorot_uniform', dtype=d_type,
                                   name="{}_lstm_cell".format(name))
    elif cell_type == 'gru':
        rnn = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True,
                                  recurrent_initializer='glorot_uniform', dtype=d_type, name="{}_gru_cell".format(name))
    else:
        print('cell执行了类型执行出错，定位细节参见log')
        log_operator(level=10).info("cell执行了类型执行出错")

    if if_bidirectional:
        rnn = tf.keras.layers.Bidirectional(layer=rnn, dtype=d_type, name="{}_biRnn".format(name))

    rnn_outputs = rnn(inputs)
    outputs = rnn_outputs[0]
    states = outputs[:, -1, :]

    return tf.keras.Model(inputs=inputs, outputs=[outputs, states])


def encoder(vocab_size: int, embedding_dim: int, enc_units: int,
            num_layers: int, cell_type: str, if_bidirectional: bool = True,
            d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """
    :param vocab_size: 词汇量大小
    :param embedding_dim: 词嵌入维度
    :param enc_units: 单元大小
    :param num_layers: encoder中内部RNN层数
    :param cell_type: cell类型，lstm/gru， 默认lstm
    :param if_bidirectional: 是否双向
    :param d_type: 运算精度
    :param name: 名称
    :return: Seq2Seq的Encoder
    """
    inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_inputs".format(name))
    outputs = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                        dtype=d_type, name="{}_embedding".format(name))(inputs)

    for i in range(num_layers):
        outputs, states = rnn_layer(
            units=enc_units, input_feature_dim=outputs.shape[-1], cell_type=cell_type,
            d_type=d_type, if_bidirectional=if_bidirectional, name="{}_rnn_{}".format(name, i)
        )(outputs)

    return tf.keras.Model(inputs=inputs, outputs=[outputs, states])


def decoder(vocab_size: int, embedding_dim: int, dec_units: int, enc_units: int, num_layers: int,
            cell_type: str, d_type: tf.dtypes.DType = tf.float32, name: str = "decoder") -> tf.keras.Model:
    """
    :param vocab_size: 词汇量大小
    :param embedding_dim: 词嵌入维度
    :param dec_units: decoder单元大小
    :param enc_units: encoder单元大小
    :param num_layers: encoder中内部RNN层数
    :param cell_type: cell类型，lstm/gru， 默认lstm
    :param d_type: 运算精度
    :param name: 名称
    :return: Seq2Seq的Decoder
    """
    inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_inputs".format(name))
    enc_output = tf.keras.Input(shape=(None, enc_units), dtype=d_type, name="{}_enc_output".format(name))
    dec_hidden = tf.keras.Input(shape=(enc_units,), dtype=d_type, name="{}_dec_hidden".format(name))

    embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                           dtype=d_type, name="{}_embedding".format(name))(inputs)
    context_vector, attention_weight = bahdanau_attention(
        units=dec_units, d_type=d_type, query_dim=enc_units, value_dim=enc_units)(inputs=[dec_hidden, enc_output])
    outputs = tf.concat(values=[tf.expand_dims(input=context_vector, axis=1), embeddings], axis=-1)

    for i in range(num_layers):
        # Decoder中不允许使用双向
        outputs, states = rnn_layer(units=dec_units, input_feature_dim=outputs.shape[-1], cell_type=cell_type,
                                    if_bidirectional=False, d_type=d_type, name="{}_rnn_{}".format(name, i))(outputs)

    outputs = tf.reshape(tensor=outputs, shape=(-1, outputs.shape[-1]))
    outputs = tf.keras.layers.Dense(units=vocab_size, dtype=d_type, name="{}_outputs_dense".format(name))(outputs)

    return tf.keras.Model(inputs=[inputs, enc_output, dec_hidden], outputs=[outputs, states, attention_weight])
