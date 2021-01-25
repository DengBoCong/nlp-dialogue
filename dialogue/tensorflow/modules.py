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
import tensorflow as tf
from dialogue.tensorflow.load_dataset import load_data
from dialogue.tools import get_dict_string
from dialogue.tools import ProgressBar


class Modules(abc.ABC):
    def __init__(self, loss_metric: tf.keras.metrics.Mean, accuracy_metric: tf.keras.metrics.SparseCategoricalAccuracy,
                 batch_size: int, buffer_size: int, max_sentence: int, train_data_type: str, valid_data_type: str,
                 dict_path: str = "", model: tf.keras.Model = None, encoder: tf.keras.Model = None,
                 decoder: tf.keras.Model = None) -> None:
        """model以及(encoder，decoder)两类模型传其中一种即可，具体在各自继承之后的训练步中使用

        :param loss_metric: 损失计算器
        :param accuracy_metric: 精度计算器
        :param batch_size: Dataset加载批大小
        :param buffer_size: Dataset加载缓存大小
        :param max_sentence: 最大句子长度
        :param train_data_type: 读取训练数据类型，单轮/多轮...
        :param valid_data_type: 读取验证数据类型，单轮/多轮...
        :param dict_path: 字典路径，若使用phoneme则不用传
        :param model: 模型
        :param encoder: encoder模型
        :param decoder: decoder模型
        :return: 返回历史指标数据
        """
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.max_sentence = max_sentence
        self.train_data_type = train_data_type
        self.valid_data_type = valid_data_type
        self.dict_path = dict_path
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    @abc.abstractmethod
    def _train_step(self, batch_dataset: tuple, optimizer: tf.optimizers.Adam, train_loss: tf.keras.metrics.Mean,
                    train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy, *args, **kwargs) -> dict:
        """该方法用于定于训练步中，模型实际训练的核心代码（在train方法中使用）

        Note:
            a): 返回所得指标字典
            b): 模型训练指标中，损失器和精度器必传，保证至少返回到当前batch为止的平均训练损失和训练精度
            c): 参数如batch_dataset、optimizer、model、encoder、decoder 为模型训练必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _valid_step(self, dataset: tf.data.Dataset, valid_loss: tf.keras.metrics.Mean, steps_per_epoch: int,
                    batch_size: int, valid_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
                    *args, **kwargs) -> dict:
        """ 该方法用于定义验证模型逻辑

        Note:
            a): 返回所得指标字典
            b): 模型验证指标中，损失器和精度器必传，保证至少返回验证的平均验证损失和验证精度
            c): dataset、model、encoder、decoder 为模型训练必需
        """

        raise NotImplementedError("Must be implemented in subclasses.")

    def train(self, optimizer: tf.optimizers.Adam, checkpoint: tf.train.CheckpointManager, train_data_path: str,
              epochs: int, checkpoint_save_freq: int, valid_data_split: float = 0.0, max_train_data_size: int = 0,
              valid_data_path: str = "", max_valid_data_size: int = 0, history: dict = {}, **kwargs) -> dict:
        """ 训练模块

        :param optimizer: 优化器
        :param checkpoint: 检查点管理器
        :param train_data_path: 文本数据路径
        :param epochs: 训练周期
        :param checkpoint_save_freq: 检查点保存频率
        :param valid_data_split: 用于从训练数据中划分验证数据
        :param max_train_data_size: 最大训练数据量
        :param valid_data_path: 验证数据文本路径
        :param max_valid_data_size: 最大验证数据量
        :param history: 用于保存训练过程中的历史指标数据
        :return: 返回历史指标数据
        """
        print('训练开始，正在准备数据中')
        train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = load_data(
            dict_path=self.dict_path, train_data_path=train_data_path, buffer_size=self.buffer_size,
            batch_size=self.batch_size, max_sentence=self.max_sentence, valid_data_split=valid_data_split,
            valid_data_path=valid_data_path, max_train_data_size=max_train_data_size,
            valid_data_type=self.valid_data_type, max_valid_data_size=max_valid_data_size,
            train_data_type=self.train_data_type, **kwargs
        )

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            train_metrics = {}
            self.loss_metric.reset_states()
            self.accuracy_metric.reset_states()

            progress_bar = ProgressBar(total=steps_per_epoch, num=self.batch_size)

            for (batch, batch_dataset) in enumerate(train_dataset.take(steps_per_epoch)):
                train_metrics = self._train_step(
                    batch_dataset=batch_dataset, optimizer=optimizer,
                    train_loss=self.loss_metric, train_accuracy=self.accuracy_metric, **kwargs
                )

                progress_bar(current=batch + 1, metrics=get_dict_string(train_metrics))

            progress_bar.done(step_time=time.time() - start_time)

            for key, value in train_metrics.items():
                history[key].append(value.numpy())

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save()

                if valid_steps_per_epoch == 0 or valid_dataset is None:
                    print("验证数据量过小，小于batch_size，已跳过验证轮次")
                else:
                    valid_metrics = self._valid_step(
                        dataset=valid_dataset, batch_size=self.batch_size, valid_accuracy=self.accuracy_metric,
                        valid_loss=self.loss_metric, steps_per_epoch=valid_steps_per_epoch, **kwargs
                    )

                    for key, value in valid_metrics.items():
                        history[key].append(value.numpy())

        print('训练结束')
        return history

    def evaluate(self, valid_data_path: str = "", max_valid_data_size: int = 0, **kwargs) -> None:
        """ 验证模块

        :param valid_data_path: 验证数据文本路径
        :param max_valid_data_size: 最大验证数据量
        :return: 返回历史指标数据
        """
        print('训练开始，正在准备数据中')
        valid_dataset, _, valid_steps_per_epoch, _ = load_data(
            dict_path=self.dict_path, train_data_path=valid_data_path, valid_data_type=self.valid_data_type,
            buffer_size=self.buffer_size, train_data_type=self.train_data_type, batch_size=self.batch_size,
            max_sentence=self.max_sentence, max_train_data_size=max_valid_data_size, **kwargs
        )

        _ = self._valid_step(dataset=valid_dataset, batch_size=self.batch_size, valid_accuracy=self.accuracy_metric,
                             valid_loss=self.loss_metric, steps_per_epoch=valid_steps_per_epoch, **kwargs)

        print('验证结束')

    @abc.abstractmethod
    def inference(self, *args, **kwargs) -> str:
        """ 对话推断模块
        """

        raise NotImplementedError("Must be implemented in subclasses.")
