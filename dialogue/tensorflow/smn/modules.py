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
"""smn的模型功能实现，包含train模式、evaluate模式、chat模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pysolr
import tensorflow as tf
from dialogue.metrics import recall_at_position_k_in_n
from dialogue.tensorflow.modules import Modules
from dialogue.tensorflow.utils import get_tf_idf_top_k
from dialogue.tensorflow.utils import load_tokenizer
from dialogue.tools import get_dict_string
from dialogue.tools import ProgressBar


class SMNModule(Modules):
    def __init__(self, loss_metric: tf.keras.metrics.Mean, accuracy_metric: tf.keras.metrics.SparseCategoricalAccuracy,
                 batch_size: int, buffer_size: int, max_sentence: int, train_data_type: str, valid_data_type: str,
                 dict_path: str = "", model: tf.keras.Model = None, encoder: tf.keras.Model = None,
                 decoder: tf.keras.Model = None):
        super(SMNModule, self).__init__(
            loss_metric=loss_metric, accuracy_metric=accuracy_metric, train_data_type=train_data_type,
            valid_data_type=valid_data_type, batch_size=batch_size, buffer_size=buffer_size, max_sentence=max_sentence,
            dict_path=dict_path, model=model, encoder=encoder, decoder=decoder
        )

    @tf.function(autograph=True)
    def _train_step(self, batch_dataset: tuple, optimizer: tf.optimizers.Adam, *args, **kwargs) -> dict:
        """训练步

        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        utterances, response, label = batch_dataset
        with tf.GradientTape() as tape:
            scores = self.model(inputs=[utterances, response])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(label, scores)

        self.loss_metric(loss)
        self.accuracy_metric(label, scores)
        gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"train_loss": self.loss_metric.result(), "train_accuracy": self.accuracy_metric.result()}

    def _valid_step(self, dataset: tf.data.Dataset, steps_per_epoch: int,
                    progress_bar: ProgressBar, *args, **kwargs) -> dict:
        """ 验证步

        :param dataset: 验证步的dataset
        :param valid_loss: 损失计算器
        :param steps_per_epoch: 验证总步数
        :param batch_size: batch大小
        :param valid_accuracy: 精度计算器
        :return: 返回所得指标字典
        """
        print("验证轮次")
        start_time = time.time()
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()
        progress_bar = ProgressBar(total=steps_per_epoch, num=self.batch_size)

        scores = tf.constant([], dtype=self.model.dtype)
        labels = tf.constant([], dtype=self.model.dtype)
        for (batch, (utterances, responses, label)) in enumerate(dataset.take(steps_per_epoch)):
            score = self._valid_ont_step(utterances=utterances, responses=responses, label=label)
            scores = tf.concat(values=[scores, score[:, 1]], axis=0)
            labels = tf.concat(values=[labels, tf.cast(x=label, dtype=self.model.dtype)], axis=0)

            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(self.loss_metric.result(), self.accuracy_metric.result()))

        rn_k = recall_at_position_k_in_n(labels=[scores.numpy(), labels.numpy()], k=[1, 2, 5], n=10, tar=1.0)
        message = {
            "train_loss": self.loss_metric.result(), "train_accuracy": self.accuracy_metric.result(),
            "valid_R10@1": rn_k[0], "valid_R10@2": rn_k[1], "valid_R10@5": rn_k[2]
        }

        progress_bar(current=steps_per_epoch, metrics=get_dict_string(data=message))
        progress_bar.done(step_time=time.time() - start_time)

        return message

    @tf.function(autograph=True)
    def _valid_ont_step(self, utterances: tf.Tensor, responses: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """ 单个验证步

        :param utterances: 对话列表
        :param responses: 响应回复
        :param label: ground-true标签
        :return: 计算所得分数
        """

        score = self.model(inputs=[utterances, responses])
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(label, score)

        self.loss_metric(loss)
        self.accuracy_metric(label, score)

        return score

    def inference(self, request: list, solr: pysolr.Solr, max_utterance: int,
                  d_type: tf.dtypes.DType = tf.float32, *args, **kwargs) -> str:
        """ 对话推断模块

        :param request: 输入对话历史
        :param solr: solr服务
        :param max_utterance: 每轮最大语句数
        :param d_type: 运算精度
        :return: 返回历史指标数据
        """
        tokenizer = load_tokenizer(self.dict_path)

        history = request[-max_utterance:]
        pad_sequences = [0] * self.max_sentence
        utterance = tokenizer.texts_to_sequences(history)
        utterance_len = len(utterance)

        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != max_utterance:
            utterance = [pad_sequences] * (max_utterance - utterance_len) + utterance
        utterance = tf.keras.preprocessing.sequence.pad_sequences(sequences=utterance,
                                                                  maxlen=self.max_sentence, padding="post")

        tf_idf = get_tf_idf_top_k(history=history, k=5)
        query = "{!func}sum("
        for key in tf_idf:
            query += "product(idf(utterance," + key + "),tf(utterance," + key + ")),"
        query += ")"
        candidates = solr.search(q=query, start=0, rows=10).docs
        candidates = [candidate["utterance"][0] for candidate in candidates]

        if candidates is None:
            return "Sorry! I didn't hear clearly, can you say it again?"
        else:
            utterances = [utterance] * len(candidates)
            responses = tokenizer.texts_to_sequences(candidates)
            responses = tf.keras.preprocessing.sequence.pad_sequences(sequences=responses,
                                                                      maxlen=self.max_sentence, padding="post")

            utterances = tf.convert_to_tensor(value=utterances)
            responses = tf.convert_to_tensor(value=responses)
            index = self._inference_one_step(utterances=utterances, responses=responses)

            return candidates[index]

    @tf.function(autograph=True)
    def _inference_one_step(self, utterances: tf.Tensor, responses: tf.Tensor) -> tf.Tensor:
        """ 单个推断步

        :param utterances: 对话列表
        :param responses: 响应回复
        :return: 最大概率响应索引
        """

        scores = self.model(inputs=[utterances, responses])
        index = tf.argmax(input=scores[:, 1])

        return index
