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
"""应用于server的加载模型推断组件
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf
from dialogue.tensorflow.seq2seq.modules import Seq2SeqModule
from dialogue.tensorflow.smn.modules import SMNModule
from dialogue.tensorflow.transformer.modules import TransformerModule


def load_transformer(config_path: str) -> TransformerModule:
    """加载Transformer的Modules

    :param config_path:
    :return: TransformerModule
    """
    options, work_path = check_and_read_path(config_path=config_path)

    encoder = tf.keras.models.load_model(filepath=(work_path + options["encoder_save_path"]).replace("\\", "/"))
    decoder = tf.keras.models.load_model(filepath=(work_path + options["decoder_save_path"]).replace("\\", "/"))

    modules = TransformerModule(max_sentence=options["max_sentence"],
                                dict_path=work_path + options["dict_path"], encoder=encoder, decoder=decoder)

    return modules


def load_seq2seq(config_path: str) -> Seq2SeqModule:
    """加载Seq2Seq的Modules

    :param config_path:
    :return: Seq2SeqModule
    """
    options, work_path = check_and_read_path(config_path=config_path)

    encoder = tf.keras.models.load_model(filepath=(work_path + options["encoder_save_path"]).replace("\\", "/"))
    decoder = tf.keras.models.load_model(filepath=(work_path + options["decoder_save_path"]).replace("\\", "/"))

    modules = Seq2SeqModule(max_sentence=options["max_sentence"],
                            dict_path=work_path + options["dict_path"], encoder=encoder, decoder=decoder)

    return modules


def load_smn(config_path: str) -> SMNModule:
    """加载Seq2Seq的Modules

    :param config_path:
    :return: Seq2SeqModule
    """
    options, work_path = check_and_read_path(config_path=config_path)

    model = tf.keras.models.load_model(filepath=(work_path + options["model_save_path"]).replace("\\", "/"))
    modules = SMNModule(max_sentence=options["max_sentence"], dict_path=work_path + options["dict_path"], model=model)

    return modules


def check_and_read_path(config_path: str) -> tuple:
    """ 检查配置文件路径及读取配置文件内容及当前工作目录

    :param config_path:
    :return: options, work_path
    """
    if config_path == "":
        print("加载失败")
        exit(0)

    with open(config_path, "r", encoding="utf-8") as config_file:
        options = json.load(config_file)

    file_path = os.path.abspath(__file__)
    work_path = file_path[:file_path.find("tensorflow")]

    return options, work_path


if __name__ == '__main__':
    load_transformer(r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\transformer.json")
