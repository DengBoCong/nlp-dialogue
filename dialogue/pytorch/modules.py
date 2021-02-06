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
"""模型功能顶层封装类，包含train、evaluate等等模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from dialogue.pytorch.load_dataset import load_data
from dialogue.pytorch.utils import save_checkpoint
from dialogue.tools import get_dict_string
from dialogue.tools import ProgressBar
from typing import AnyStr
from typing import Dict
from typing import NoReturn
from typing import Tuple


class Modules(abc.ABC):
    def __init__(self, batch_size: int, max_sentence: int, train_data_type: str, valid_data_type: str,
                 dict_path: str = "", num_workers: int = 2, model: torch.nn.Module = None,
                 encoder: torch.nn.Module = None, decoder: torch.nn.Module = None) -> NoReturn:
        """model以及(encoder，decoder)两类模型传其中一种即可，具体在各自继承之后的训练步中使用
        Note:
            a): 模型训练指标中，保证至少返回到当前batch为止的平均训练损失

        :param batch_size: Dataset加载批大小
        :param max_sentence: 最大句子长度
        :param train_data_type: 读取训练数据类型，单轮/多轮...
        :param valid_data_type: 读取验证数据类型，单轮/多轮...
        :param dict_path: 字典路径，若使用phoneme则不用传
        :param num_workers: 数据加载器的工作线程
        :param model: 模型
        :param encoder: encoder模型
        :param decoder: decoder模型
        :return: 返回历史指标数据
        """
        self.batch_size = batch_size
        self.max_sentence = max_sentence
        self.train_data_type = train_data_type
        self.valid_data_type = valid_data_type
        self.dict_path = dict_path
        self.num_workers = num_workers
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    @abc.abstractmethod
    def _train_step(self, batch_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    optimizer: Optimizer, *args, **kwargs) -> Dict:
        """该方法用于定于训练步中，模型实际训练的核心代码（在train方法中使用）

        Note:
            a): 返回所得指标字典
            b): batch_dataset、optimizer为模型训练必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _valid_step(self, loader: DataLoader, steps_per_epoch: int,
                    progress_bar: ProgressBar, *args, **kwargs) -> Dict:
        """ 该方法用于定义验证模型逻辑

        Note:
            a): 返回所得指标字典
            b): DataLoader为模型验证必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def train(self, optimizer: torch.optim.Optimizer, train_data_path: str, epochs: int, checkpoint_save_freq: int,
              checkpoint_dir: str = "", valid_data_split: float = 0.0, max_train_data_size: int = 0,
              valid_data_path: str = "", max_valid_data_size: int = 0, history: dict = {}, **kwargs) -> Dict:
        """ 训练模块

        :param optimizer: 优化器
        :param train_data_path: 文本数据路径
        :param epochs: 训练周期
        :param checkpoint_save_freq: 检查点保存频率
        :param checkpoint_dir: 检查点保存目录路径
        :param valid_data_split: 用于从训练数据中划分验证数据
        :param max_train_data_size: 最大训练数据量
        :param valid_data_path: 验证数据文本路径
        :param max_valid_data_size: 最大验证数据量
        :param history: 用于保存训练过程中的历史指标数据
        :return: 返回历史指标数据
        """
        print('训练开始，正在准备数据中...')
        train_loader, valid_loader, train_steps_per_epoch, valid_steps_per_epoch = load_data(
            dict_path=self.dict_path, batch_size=self.batch_size, train_data_type=self.train_data_type,
            valid_data_type=self.valid_data_type, max_sentence=self.max_sentence, valid_data_split=valid_data_split,
            train_data_path=train_data_path, valid_data_path=valid_data_path, max_train_data_size=max_train_data_size,
            max_valid_data_size=max_valid_data_size, num_workers=self.num_workers, **kwargs
        )

        progress_bar = ProgressBar()

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            start_time = time.time()

            progress_bar.reset(total=train_steps_per_epoch, num=self.batch_size)

            for (batch, batch_dataset) in enumerate(train_loader):
                train_metrics = self._train_step(batch_dataset=batch_dataset, optimizer=optimizer, **kwargs)

                progress_bar(current=batch + 1, metrics=get_dict_string(data=train_metrics))

            progress_bar.done(step_time=time.time() - start_time)

            for key, value in train_metrics.items():
                history[key].append(value)

            if (epoch + 1) % checkpoint_save_freq == 0:
                save_checkpoint(checkpoint_dir=checkpoint_dir, optimizer=optimizer,
                                model=self.model, encoder=self.encoder, decoder=self.decoder)

                if valid_steps_per_epoch == 0 or valid_loader is None:
                    print("验证数据量过小，小于batch_size，已跳过验证轮次")
                else:
                    valid_metrics = self._valid_step(loader=valid_loader, progress_bar=progress_bar,
                                                     steps_per_epoch=valid_steps_per_epoch, **kwargs)

                    for key, value in valid_metrics.items():
                        history[key].append(value)

        print("训练结束")
        return history

    def evaluate(self, valid_data_path: str = "", max_valid_data_size: int = 0, **kwargs) -> NoReturn:
        """ 验证模块

        :param valid_data_path: 验证数据文本路径
        :param max_valid_data_size: 最大验证数据量
        :return: 返回历史指标数据
        """
        print("验证开始，正在准备数据中")
        _, valid_loader, _, valid_steps_per_epoch = load_data(
            dict_path=self.dict_path, batch_size=self.batch_size, train_data_type=self.train_data_type,
            valid_data_type=self.valid_data_type, max_sentence=self.max_sentence, valid_data_path=valid_data_path,
            max_valid_data_size=max_valid_data_size, num_workers=self.num_workers, **kwargs
        )

        progress_bar = ProgressBar()
        _ = self._valid_step(loader=valid_loader, progress_bar=progress_bar,
                             steps_per_epoch=valid_steps_per_epoch, **kwargs)

        print("验证结束")

    @abc.abstractmethod
    def inference(self, *args, **kwargs) -> AnyStr:
        """ 对话推断模块
        """

        raise NotImplementedError("Must be implemented in subclasses.")
