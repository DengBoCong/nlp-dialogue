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
"""Dataset加载模块，内含各模型针对性的以及公用性的数据加载方法
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dialogue.tools.read_data import read_data
from dialogue.tools import load_tokenizer
from typing import Tuple


def load_data(dict_path: str, buffer_size: int, batch_size: int, train_data_type: str, valid_data_type: str,
              max_sentence: int, valid_data_split: float = 0.0, train_data_path: str = "", valid_data_path: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0, **kwargs) -> Tuple:
    """ 数据加载方法

    :param dict_path: 字典路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param train_data_type: 读取训练数据类型，单轮/多轮...
    :param valid_data_type: 读取验证数据类型，单轮/多轮...
    :param max_sentence: 单个句子最大长度
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param train_data_path: 文本数据路径
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :return: 训练Dataset、验证Dataset、训练数据总共的步数、验证数据总共的步数和检查点前缀
    """
    tokenizer = load_tokenizer(dict_path=dict_path)

    train_flag = True  # 是否开启训练标记
    train_steps_per_epoch = 0
    train_first, train_second, train_third = None, None, None

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0
    valid_first, valid_second, valid_third = None, None, None

    if train_data_path != "":
        train_first, train_second, train_third = read_data(
            data_path=train_data_path, max_data_size=max_train_data_size,
            max_sentence=max_sentence, data_type=train_data_type, tokenizer=tokenizer, **kwargs
        )
    else:
        train_flag = False

    if valid_data_path != "":
        print("读取验证对话对...")
        valid_first, valid_second, valid_third = read_data(
            data_path=valid_data_path, max_data_size=max_valid_data_size,
            max_sentence=max_sentence, data_type=valid_data_type, tokenizer=tokenizer, **kwargs
        )
    elif valid_data_split != 0.0:
        train_size = int(len(train_first) * (1.0 - valid_data_split))
        valid_first = train_first[train_size:]
        valid_second = train_second[train_size:]
        valid_third = train_third[train_size:]
        train_first = train_first[:train_size]
        train_second = train_second[:train_size]
        train_third = train_third[:train_size]
    else:
        valid_flag = False

    if train_flag:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_first, train_second, train_third)).cache().shuffle(
            buffer_size, reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        train_steps_per_epoch = len(train_first) // batch_size
    else:
        train_dataset = None

    if valid_flag:
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_first, valid_second, valid_third)) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
        valid_steps_per_epoch = len(valid_first) // batch_size
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, train_steps_per_epoch, valid_steps_per_epoch
