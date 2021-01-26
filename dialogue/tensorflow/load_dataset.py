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

import os
import tensorflow as tf
from dialogue.tensorflow.utils import load_tokenizer


def load_data(dict_path: str, buffer_size: int, batch_size: int, train_data_type: str, valid_data_type: str,
              max_sentence: int, valid_data_split: float = 0.0, train_data_path: str = "", valid_data_path: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0, **kwargs) -> tuple:
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
        train_first, train_second, train_third = _read_data(
            data_path=train_data_path, max_data_size=max_train_data_size,
            max_sentence=max_sentence, data_type=train_data_type, tokenizer=tokenizer, **kwargs
        )
    else:
        train_flag = False

    if valid_data_path != "":
        print("读取验证对话对...")
        valid_first, valid_second, valid_third = _read_data(
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


def _read_data(data_path: str, max_data_size: int, max_sentence: int, data_type: str,
               tokenizer: tf.keras.preprocessing.text.Tokenizer, **kwargs) -> tuple:
    """ 中转读取数据

    :param data_path: 分词文本路径
    :param max_data_size: 读取的数据量大小
    :param max_sentence: 最大序列长度
    :param data_type: 读取数据类型，单轮/多轮
    :param tokenizer: 传入现有的分词器，默认重新生成
    :return: 输入序列张量、目标序列张量和分词器
    """
    operation = {
        "read_single_data": lambda: _read_single_data(
            data_path=data_path, max_data_size=max_data_size, max_sentence=max_sentence, tokenizer=tokenizer),
        "read_multi_turn_data": lambda: _read_multi_turn_data(
            data_path=data_path, max_data_size=max_data_size, max_utterance=kwargs.get("max_utterance"),
            max_sentence=max_sentence, tokenizer=tokenizer)
    }

    return operation.get(data_type)()


def _read_single_data(data_path: str, max_data_size: int, max_sentence: int,
                      tokenizer: tf.keras.preprocessing.text.Tokenizer) -> tuple:
    """ 读取单轮问答数据，将input和target进行分词后，与样本权重一同返回

    :param data_path: 分词文本路径
    :param max_data_size: 读取的数据量大小
    :param max_sentence: 最大序列长度
    :param tokenizer: 传入现有的分词器，默认重新生成
    :return: 输入序列张量、目标序列张量和分词器
    """
    if not os.path.exists(data_path):
        print("不存在已经分词好的文件，请检查数据集或执行pre_treat模式")
        exit(0)

    with open(data_path, "r", encoding="utf-8") as file:
        sample_weights = []
        qa_pairs = []
        count = 0  # 用于处理数据计数

        for line in file:
            # 文本数据中的问答对权重通过在问答对尾部添加“<|>”配置
            temp = line.strip().strip("\n").replace('/', '').split("<|>")
            qa_pairs.append([sentence for sentence in temp[0].split("\t")])
            # 如果没有配置对应问答对权重，则默认为1.
            if len(temp) == 1:
                sample_weights.append(float(1))
            else:
                sample_weights.append(float(temp[1]))

            count += 1
            if max_data_size == count:
                break

    (input_lang, target_lang) = zip(*qa_pairs)

    input_tensor = tokenizer.texts_to_sequences(input_lang)
    target_tensor = tokenizer.texts_to_sequences(target_lang)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_sentence, padding="post")
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_sentence, padding="post")

    return input_tensor, target_tensor, sample_weights


def _read_multi_turn_data(data_path: str, max_data_size: int, max_utterance: int, max_sentence: int,
                          tokenizer: tf.keras.preprocessing.text.Tokenizer) -> tuple:
    """ 读取多轮对话数据，将utterance和response进行分词后，同label等数据一并返回

    :param data_path: 分词文本路径
    :param max_data_size: 读取的数据量大小
    :param max_utterance: 每轮对话最大对话数
    :param max_sentence: 单个句子最大长度
    :param tokenizer: 传入现有的分词器，默认重新生成
    :return: 输入序列张量、目标序列张量和分词器
        """
    if not os.path.exists(data_path):
        print("不存在已经分词好的文件，请检查数据集或执行pre_treat模式")
        exit(0)

    history = []  # 用于保存每轮对话历史语句
    response = []  # 用于保存每轮对话的回答
    label = []  # 用于保存每轮对话的标签
    count = 0  # 用于处理数据计数

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            apart = line.strip().strip("\n").replace('/', '').split('\t')
            label.append(int(apart[0]))
            response.append(apart[-1])
            del apart[0]
            del apart[-1]
            history.append(apart)

            count += 1
            if max_data_size == count:
                break

    response = tokenizer.texts_to_sequences(response)
    response = tf.keras.preprocessing.sequence.pad_sequences(response, maxlen=max_sentence, padding="post")

    utterances = []
    for utterance in history:
        # 注意了，这边要取每轮对话的最后max_utterances数量的语句
        utterance_padding = tokenizer.texts_to_sequences(utterance)[-max_utterance:]
        utterance_len = len(utterance_padding)
        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != max_utterance:
            utterance_padding += [[0]] * (max_utterance - utterance_len)
        utterances.append(tf.keras.preprocessing.sequence.pad_sequences(
            utterance_padding, maxlen=max_sentence, padding="post").tolist())

    return utterances, response, label
