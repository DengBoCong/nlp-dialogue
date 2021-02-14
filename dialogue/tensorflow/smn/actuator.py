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
"""smn结构的实现执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pysolr
import tensorflow as tf
from argparse import ArgumentParser
from dialogue.tensorflow.preprocess import create_search_data
from dialogue.tensorflow.smn.model import smn
from dialogue.tensorflow.smn.modules import SMNModule
from dialogue.tensorflow.utils import load_checkpoint
from typing import NoReturn


def tf_smn() -> NoReturn:
    parser = ArgumentParser(description="smn chatbot")
    parser.add_argument("--version", default="tf", type=str, required=True, help="执行版本")
    parser.add_argument("--model", default="transformer", type=str, required=True, help="执行模型")
    parser.add_argument("--config_file", default="", type=str, required=False, help="配置文件路径，为空则默认命令行，不为空则使用配置文件参数")
    parser.add_argument("--act", default="pre_treat", type=str, required=False, help="执行类型")
    parser.add_argument("--units", default=200, type=int, required=False, help="隐藏层单元数")
    parser.add_argument("--vocab_size", default=200000, type=int, required=False, help="词汇大小")
    parser.add_argument("--embedding_dim", default=512, type=int, required=False, help="嵌入层维度大小")
    parser.add_argument("--max_sentence", default=50, type=int, required=False, help="单个句子序列最大长度")
    parser.add_argument("--max_utterance", default=10, type=int, required=False, help="当回合最大句子数")
    parser.add_argument("--max_train_data_size", default=0, type=int, required=False, help="用于训练的最大数据大小")
    parser.add_argument("--max_valid_data_size", default=0, type=int, required=False, help="用于验证的最大数据大小")
    parser.add_argument("--checkpoint_save_freq", default=2, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--checkpoint_save_size", default=1, type=int, required=False, help="单轮训练中检查点保存数量")
    parser.add_argument("--valid_data_split", default=0.0, type=float, required=False, help="从训练数据集中划分验证数据的比例")
    parser.add_argument("--learning_rate", default=0.001, type=float, required=False, help="学习率")
    parser.add_argument("--max_database_size", default=0, type=int, required=False, help="最大数据候选数量")
    parser.add_argument("--dict_path", default="data\\preprocess\\smn_dict.json", type=str, required=False, help="字典路径")
    parser.add_argument("--checkpoint_dir", default="checkpoints\\tensorflow\\smn", type=str, required=False,
                        help="检查点路径")
    parser.add_argument("--train_data_path", default="data\\ubuntu_train.txt", type=str, required=False,
                        help="处理好的多轮分词训练数据集路径")
    parser.add_argument("--valid_data_path", default="data\\ubuntu_valid.txt", type=str, required=False,
                        help="处理好的多轮分词验证数据集路径")
    parser.add_argument("--solr_server", default="http://49.235.33.100:8983/solr/smn/", type=str, required=False,
                        help="solr服务地址")
    parser.add_argument("--candidate_database", default="data\\preprocess\\candidate.json", type=str, required=False,
                        help="候选回复数据库")
    parser.add_argument("--epochs", default=5, type=int, required=False, help="训练步数")
    parser.add_argument("--batch_size", default=64, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=20000, type=int, required=False, help="Dataset加载缓冲大小")
    parser.add_argument("--start_sign", default="<start>", type=str, required=False, help="序列开始标记")
    parser.add_argument("--end_sign", default="<end>", type=str, required=False, help="序列结束标记")
    parser.add_argument("--unk_sign", default="<unk>", type=str, required=False, help="未登录词")
    parser.add_argument("--model_save_path", default="models\\tensorflow\\smn", type=str,
                        required=False, help="SaveModel格式保存路径")

    options = parser.parse_args().__dict__
    execute_type = options["act"]
    if options["config_file"] != "":
        with open(options["config_file"], "r", encoding="utf-8") as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以tensorflow目录下为基准配置
    file_path = os.path.abspath(__file__)
    work_path = file_path[:file_path.find("tensorflow")]

    model = smn(units=options["units"], vocab_size=options["vocab_size"], embedding_dim=options["embedding_dim"],
                max_utterance=options["max_utterance"], max_sentence=options["max_sentence"])

    loss_metric = tf.keras.metrics.Mean(name="loss_metric")
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy_metric")
    optimizer = tf.optimizers.Adam(learning_rate=options["learning_rate"], name="optimizer")
    checkpoint_manager = load_checkpoint(
        checkpoint_dir=work_path + options["checkpoint_dir"], execute_type=execute_type,
        model=model, checkpoint_save_size=options["checkpoint_save_size"]
    )

    modules = SMNModule(
        loss_metric=loss_metric, accuracy_metric=accuracy_metric, batch_size=options["batch_size"],
        buffer_size=options["buffer_size"], max_sentence=options["max_sentence"],
        train_data_type="read_multi_turn_data", valid_data_type="read_multi_turn_data",
        dict_path=work_path + options["dict_path"], model=model
    )

    if execute_type == "pre_treat":
        create_search_data(data_path=work_path + options["train_data_path"], solr_server=options["solr_server"],
                           max_database_size=options["max_database_size"], vocab_size=options["vocab_size"],
                           dict_path=work_path + options["dict_path"], unk_sign=options["unk_sign"])
    elif execute_type == "train":
        history = {
            "train_accuracy": [], "train_loss": [], "valid_accuracy": [],
            "valid_loss": [], "valid_R10@1": [], "valid_R10@2": [], "valid_R10@5": []
        }
        history = modules.train(
            optimizer=optimizer, epochs=options["epochs"], checkpoint=checkpoint_manager, history=history,
            train_data_path=work_path + options["train_data_path"], max_utterance=options["max_utterance"],
            checkpoint_save_freq=options["checkpoint_save_freq"], max_valid_data_size=options["max_valid_data_size"],
            max_train_data_size=options["max_train_data_size"], valid_data_split=options["valid_data_split"],
            valid_data_path=work_path + options["valid_data_path"],
            model_save_path=work_path + options["model_save_path"]
        )
        # show_history(history=history, valid_freq=options["checkpoint_save_freq"],
        #              save_dir=work_path + options["history_image_dir"])
    elif execute_type == "evaluate":
        modules.evaluate(
            max_valid_data_size=options["max_valid_data_size"], valid_data_path=work_path + options["valid_data_path"],
            max_utterance=options["max_utterance"]
        )
    elif execute_type == "chat":
        history = []  # 用于存放历史对话
        solr = pysolr.Solr(url=options["solr_server"], always_commit=True, timeout=10)

        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            request = input("User: ")
            if request == "ESC":
                print("Agent: 再见！")
                exit(0)
            history.append(request)
            response = modules.inference(request=history, solr=solr, max_utterance=options["max_utterance"])
            history.append(response)
            print("Agent: ", response)
    else:
        parser.error(message="")


if __name__ == "__main__":
    tf_smn()
