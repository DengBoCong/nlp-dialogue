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
import tensorflow as tf
from dialogue.metrics import recall_at_position_k_in_n
from dialogue.tensorflow.beamsearch import BeamSearch
from dialogue.tensorflow.modules import Modules
from dialogue.tensorflow.utils import load_tokenizer
from dialogue.tensorflow.utils import preprocess_request
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

    def _train_step(self, dataset: tf.data.Dataset, steps_per_epoch: int,
                    progress_bar: ProgressBar, optimizer: tf.optimizers.Adam, *args, **kwargs) -> dict:
        """训练步

        :param dataset: 训练步的dataset
        :param steps_per_epoch: 训练总步数
        :param progress_bar: 进度管理器
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        start_time = time.time()
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()
        progress_bar.reset(total=steps_per_epoch, num=self.batch_size)

        for (batch, (utterances, response, label)) in enumerate(dataset.take(steps_per_epoch)):
            with tf.GradientTape() as tape:
                scores = self.model(inputs=[utterances, response])
                loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)(label, scores)

            self.loss_metric(loss)
            self.accuracy_metric(label, scores)
            gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(self.loss_metric.result(), self.accuracy_metric.result()))

        progress_bar.done(step_time=time.time() - start_time)

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
        for (batch, (utterances, response, label)) in enumerate(dataset.take(steps_per_epoch)):
            print(label)
            exit(0)
            score = self.model(inputs=[utterances, response])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)(label, score)

            self.loss_metric(loss)
            self.accuracy_metric(label, score)
            scores = tf.concat(values=[scores, score[:, 1]], axis=0)
            labels = tf.concat(values=[labels, tf.cast(x=label, dtype=self.model.dtype)], axis=0)

            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(self.loss_metric.result(), self.accuracy_metric.result()))

        # scores_label = tf.concat(
        #     values=[tf.expand_dims(input=scores, axis=1), tf.expand_dims(input=labels, axis=1)], axis=1
        # ).numpy()

        rn_k = recall_at_position_k_in_n(labels=[scores, labels], k=[1, 2, 5], n=10, tar=1.0)
        print(rn_k)
        exit(0)

        progress_bar.done(step_time=time.time() - start_time)

        return {"valid_loss": valid_loss.result(), "valid_accuracy": valid_accuracy.result()}

    def inference(self, request: str, beam_size: int, start_sign: str = "<start>", end_sign: str = "<end>") -> str:
        """ 对话推断模块

        :param request: 输入句子
        :param beam_size: beam大小
        :param start_sign: 句子开始标记
        :param end_sign: 句子结束标记
        :return: 返回历史指标数据
        """
        tokenizer = load_tokenizer(self.dict_path)

        enc_input = preprocess_request(sentence=request, tokenizer=tokenizer,
                                       max_length=self.max_length, start_sign=start_sign, end_sign=end_sign)
        enc_output, states = self.encoder(inputs=enc_input)
        dec_input = tf.expand_dims([tokenizer.word_index.get(start_sign)], 0)

        beam_search_container = BeamSearch(beam_size=beam_size, max_length=self.max_length, worst_score=0)
        beam_search_container.reset(enc_output=enc_output, dec_input=dec_input, remain=states)
        enc_output, dec_input, states = beam_search_container.get_search_inputs()

        for t in range(self.max_length):
            predictions, _, _ = self.decoder(inputs=[dec_input, enc_output, states])
            predictions = tf.nn.softmax(predictions, axis=-1)

            beam_search_container.expand(predictions=predictions, end_sign=tokenizer.word_index.get(end_sign))
            if beam_search_container.beam_size == 0:
                break

            enc_output, dec_input, states = beam_search_container.get_search_inputs()
            dec_input = tf.expand_dims(input=dec_input[:, -1], axis=-1)

        beam_search_result = beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = tokenizer.sequences_to_texts(temp)
            text[0] = text[0].replace(start_sign, '').replace(end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
