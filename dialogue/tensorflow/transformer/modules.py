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
"""transformer的模型功能实现，包含train模式、evaluate模式、chat模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from dialogue.tensorflow.beamsearch import BeamSearch
from dialogue.tensorflow.modules import Modules
from dialogue.tensorflow.optimizers import loss_func_mask
from dialogue.tensorflow.utils import load_tokenizer
from dialogue.tensorflow.utils import preprocess_request
from dialogue.tools import ProgressBar


class TransformerModule(Modules):
    def __init__(self, loss_metric: tf.keras.metrics.Mean, accuracy_metric: tf.keras.metrics.SparseCategoricalAccuracy,
                 batch_size: int, buffer_size: int, max_sentence: int, data_type: str, dict_path: str = "",
                 model: tf.keras.Model = None, encoder: tf.keras.Model = None, decoder: tf.keras.Model = None):
        super(TransformerModule, self).__init__(
            loss_metric=loss_metric, accuracy_metric=accuracy_metric, data_type=data_type, batch_size=batch_size,
            buffer_size=buffer_size, max_sentence=max_sentence, dict_path=dict_path, model=model, encoder=encoder,
            decoder=decoder
        )

    def _train_step(self, batch_dataset: tuple, optimizer: tf.optimizers.Adam, train_loss: tf.keras.metrics.Mean,
                    train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy, *args, **kwargs) -> dict:
        """训练步

        :param batch_dataset: batch的数据
        :param optimizer: 优化器
        :param train_loss: 损失计算器
        :param train_accuracy: 精度计算器
        :return: 返回所得指标字典
        """
        target_input = batch_dataset[1][:, :-1]
        target_real = batch_dataset[1][:, 1:]
        with tf.GradientTape() as tape:
            encoder_outputs, padding_mask = self.encoder(inputs=batch_dataset[0])
            predictions = self.decoder(inputs=[target_input, encoder_outputs, padding_mask])
            loss = loss_func_mask(target_real, predictions, batch_dataset[2])
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        train_loss(loss)
        train_accuracy(target_real, predictions)

        return {"train_loss": train_loss.result(), "train_accuracy": train_accuracy.result()}

    def _valid_step(self, dataset: tf.data.Dataset, valid_loss: tf.keras.metrics.Mean, steps_per_epoch: int,
                    batch_size: int, valid_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
                    *args, **kwargs) -> dict:
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
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        progress_bar = ProgressBar(total=steps_per_epoch, num=batch_size)

        for (batch, (inp, target, _)) in enumerate(dataset.take(steps_per_epoch)):
            target_input = target[:, :-1]
            target_real = target[:, 1:]

            encoder_outputs, padding_mask = self.encoder(inputs=inp)
            predictions = self.decoder(inputs=[target_input, encoder_outputs, padding_mask])
            loss = loss_func_mask(target_real, predictions)

            valid_loss(loss)
            valid_accuracy(target_real, predictions)

            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(valid_loss.result(), valid_accuracy.result()))

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
                                       max_length=self.max_sentence, start_sign=start_sign, end_sign=end_sign)
        enc_output, padding_mask = self.encoder(inputs=enc_input)
        dec_input = tf.expand_dims([tokenizer.word_index.get(start_sign)], 0)

        beam_search_container = BeamSearch(beam_size=beam_size, max_length=self.max_sentence, worst_score=0)
        beam_search_container.reset(enc_output=enc_output, dec_input=dec_input, remain=padding_mask)
        enc_output, dec_input, padding_mask = beam_search_container.get_search_inputs()

        for t in range(self.max_sentence):
            predictions = self.decoder(inputs=[dec_input, enc_output, padding_mask])
            predictions = tf.nn.softmax(predictions, axis=-1)
            predictions = predictions[:, -1:, :]
            predictions = tf.squeeze(predictions, axis=1)

            beam_search_container.expand(predictions=predictions, end_sign=tokenizer.word_index.get(end_sign))
            # 注意了，如果BeamSearch容器里的beam_size为0了，说明已经找到了相应数量的结果，直接跳出循环
            if beam_search_container.beam_size == 0:
                break
            enc_output, dec_input, padding_mask = beam_search_container.get_search_inputs()

        beam_search_result = beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = tokenizer.sequences_to_texts(temp)
            text[0] = text[0].replace(start_sign, '').replace(end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
