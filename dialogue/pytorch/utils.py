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
"""相关工具集
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import torch
from typing import NoReturn
from typing import Tuple


def save_checkpoint(checkpoint_dir: str, optimizer: torch.optim.Optimizer = None, model: torch.nn.Module = None,
                    encoder: torch.nn.Module = None, decoder: torch.nn.Module = None) -> NoReturn:
    """ 保存模型检查点

    :param checkpoint_dir: 检查点保存路径
    :param optimizer: 优化器
    :param model: 模型
    :param encoder: encoder模型
    :param decoder: decoder模型
    :return: 无返回值
    """
    checkpoint_path = checkpoint_dir + "checkpoint"
    version = 1
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as file:
            json_string = file.read().strip().strip("\n")
            info = json.load(json_string)
            version = info["version"] + 1

    model_dict = {}
    if model is not None:
        model_dict["model_state_dict"] = model.state_dict()
    if encoder is not None:
        model_dict["encoder_state_dict"] = encoder.state_dict()
    if decoder is not None:
        model_dict["decoder_state_dict"] = decoder.state_dict()
    model_dict["optimizer_state_dict"] = optimizer.state_dict()

    model_checkpoint_path = "checkpoint-{}.pth".format(version)
    torch.save(model_dict, checkpoint_dir + model_checkpoint_path)
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        file.write(json.dumps({
            "version": version,
            "model_checkpoint_path": model_checkpoint_path,
            "last_preserved_timestamp": time.time()
        }))


def load_checkpoint(checkpoint_dir: str, execute_type: str, optimizer: torch.optim.Optimizer = None,
                    model: torch.nn.Module = None, encoder: torch.nn.Module = None,
                    decoder: torch.nn.Module = None) -> Tuple:
    """加载检查点恢复模型，同时支持Encoder-Decoder结构加载

    :param checkpoint_dir: 检查点保存路径
    :param execute_type: 执行类型
    :param optimizer: 优化器
    :param model: 模型
    :param encoder: encoder模型
    :param decoder: decoder模型
    :return: 恢复的各模型检查点细节
    """
    checkpoint_path = checkpoint_dir + "checkpoint"

    if not os.path.exists(checkpoint_path) and execute_type != "train" and execute_type != "pre_treat":
        print("没有检查点，请先执行train模式")
        exit(0)
    elif not os.path.exists(checkpoint_path):
        return model, encoder, decoder, optimizer

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        checkpoint_info = json.load(file)

    model_checkpoint_path = checkpoint_dir + checkpoint_info["model_checkpoint_path"]

    checkpoint = torch.load(model_checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if encoder is not None:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if decoder is not None:
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, encoder, decoder, optimizer
