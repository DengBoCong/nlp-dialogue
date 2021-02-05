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
"""预处理操作，包含在正式使用模型前（训练、评估、推断等操作前）进行相关预处理
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pysolr
import tensorflow as tf
from typing import NoReturn


def create_search_data(data_path: str, solr_server: str, max_database_size: int,
                       vocab_size: int, dict_path: str, unk_sign: str = "<unk>") -> NoReturn:
    """ 生成轮次tf-idf为索引的候选回复

    :param data_path: 文本数据路径
    :param solr_server: solr服务的地址
    :param max_database_size: 从文本中读取最大数据量
    :param vocab_size: 词汇量大小
    :param dict_path: 字典保存路径
    :param unk_sign: 未登录词
    :return: 无返回值
    """
    if not os.path.exists(data_path):
        print("没有找到对应的文本数据，请确认文本数据存在")
        exit(0)

    responses = []
    all_text_list = []
    solr = pysolr.Solr(url=solr_server, always_commit=True)
    solr.ping()

    print("检测到对应文本，正在处理文本数据")
    with open(data_path, "r", encoding="utf-8") as file:
        count = 0
        odd_flag = True
        for line in file:
            odd_flag = not odd_flag
            if odd_flag:
                continue

            line = line.strip("\n").replace("/", "")
            apart = line.split("\t")[1:]
            all_text_list.extend(apart)
            for i in range(len(apart)):
                responses.append({"utterance": apart[i]})

            count += 1
            print("\r已处理了 {} 轮次对话".format(count), flush=True, end="")
            if max_database_size == count:
                break

    solr.delete(q="*:*")
    solr.add(docs=responses)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", num_words=vocab_size, oov_token=unk_sign)
    tokenizer.fit_on_texts(all_text_list)
    with open(dict_path, "w", encoding="utf-8") as dict_file:
        dict_file.write(tokenizer.to_json())

    print("\n文本处理完毕，已更新候选回复集，并且以保存字典")
