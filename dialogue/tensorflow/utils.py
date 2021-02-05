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
"""支持各类任务的工具，检查点加载、分词器加载、句子预处理、mask等等
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from typing import List


def combine_mask(seq: tf.Tensor) -> Tuple:
    """对input中的不能见单位进行mask

    :param seq: 输入序列
    :param d_type: 运算精度
    :return: mask
    """
    look_ahead_mask = _create_look_ahead_mask(seq)
    padding_mask = create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)


def create_padding_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建输入序列的扩充部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(x=tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def _create_look_ahead_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建当前点以后位置部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask


def load_checkpoint(checkpoint_dir: str, execute_type: str, checkpoint_save_size: int, model: tf.keras.Model = None,
                    encoder: tf.keras.Model = None, decoder: tf.keras.Model = None) -> tf.train.CheckpointManager:
    """加载检查点，同时支持Encoder-Decoder结构加载，两种类型的模型二者只能传其一

    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    :param model: 传入的模型
    :param encoder: 传入的Encoder模型
    :param decoder: 传入的Decoder模型
    """
    if model is not None:
        checkpoint = tf.train.Checkpoint(model=model)
    elif encoder is not None and decoder is not None:
        checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    else:
        print("加载检查点所传入模型有误，请检查后重试！")

    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir,
                                                    max_to_keep=checkpoint_save_size)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    else:
        if execute_type != "train" and execute_type != "pre_treat":
            print("没有检查点，请先执行train模式")
            exit(0)

    return checkpoint_manager


def get_tf_idf_top_k(history: list, k: int = 5) -> List:
    """ 使用tf_idf算法计算权重最高的k个词，并返回

    :param history: 上下文语句
    :param k: 返回词数量
    :return: top_5_key
    """
    tf_idf = {}

    vectorizer = TfidfVectorizer(analyzer="word")
    weights = vectorizer.fit_transform(history).toarray()[-1]
    key_words = vectorizer.get_feature_names()

    for i in range(len(weights)):
        tf_idf[key_words[i]] = weights[i]

    top_k_key = []
    tf_idf_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:k]
    for element in tf_idf_sorted:
        top_k_key.append(element[0])

    return top_k_key
