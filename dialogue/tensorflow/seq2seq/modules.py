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
"""seq2seq的模型功能实现，包含train模式、evaluate模式、chat模式
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


class Seq2SeqModule(Modules):
    def __init__(self, loss_metric: tf.keras.metrics.Mean = None, batch_size: int = 0, buffer_size: int = 0,
                 accuracy_metric: tf.keras.metrics.SparseCategoricalAccuracy = None, max_sentence: int = 0,
                 train_data_type: str = "", valid_data_type: str = "", dict_path: str = "",
                 model: tf.keras.Model = None, encoder: tf.keras.Model = None, decoder: tf.keras.Model = None):
        super(Seq2SeqModule, self).__init__(
            loss_metric=loss_metric, accuracy_metric=accuracy_metric, train_data_type=train_data_type,
            valid_data_type=valid_data_type, batch_size=batch_size, buffer_size=buffer_size, max_sentence=max_sentence,
            dict_path=dict_path, model=model, encoder=encoder, decoder=decoder
        )

    def _save_model(self, **kwargs) -> None:
        self.encoder.save(filepath=kwargs["encoder_save_path"])
        self.decoder.save(filepath=kwargs["decoder_save_path"])
        print("模型已保存为SaveModel格式")

    @tf.function(autograph=True)
    def _train_step(self, batch_dataset: tuple, optimizer: tf.optimizers.Adam, *args, **kwargs) -> dict:
        """训练步

        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        loss = 0.
        inputs, targets, weights = batch_dataset

        with tf.GradientTape() as tape:
            enc_output, states = self.encoder(inputs=inputs)
            dec_input = tf.expand_dims(input=[kwargs.get("start_sign", 2)] * self.batch_size, axis=1)
            for t in range(1, self.max_sentence):
                predictions, states, _ = self.decoder(inputs=[dec_input, enc_output, states])
                loss += loss_func_mask(real=targets[:, t], pred=predictions, weights=weights)
                self.accuracy_metric(targets[:, t], predictions)
                dec_input = tf.expand_dims(targets[:, t], 1)

        self.loss_metric(loss)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return {"train_loss": self.loss_metric.result(), "train_accuracy": self.accuracy_metric.result()}

    def _valid_step(self, dataset: tf.data.Dataset, steps_per_epoch: int,
                    progress_bar: ProgressBar, *args, **kwargs) -> dict:
        """ 验证步

        :param dataset: 验证步的dataset
        :param steps_per_epoch: 验证总步数
        :param progress_bar: 进度管理器
        :return: 返回所得指标字典
        """
        print("验证轮次")
        start_time = time.time()
        self.loss_metric.reset_states()
        self.accuracy_metric.reset_states()
        progress_bar = ProgressBar(total=steps_per_epoch, num=self.batch_size)

        for (batch, (inputs, target, _)) in enumerate(dataset.take(steps_per_epoch)):
            loss = self._valid_one_step(inputs=inputs, target=target, **kwargs)
            self.loss_metric(loss)
            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(self.loss_metric.result(), self.accuracy_metric.result()))

        progress_bar.done(step_time=time.time() - start_time)

        return {"valid_loss": self.loss_metric.result(), "valid_accuracy": self.accuracy_metric.result()}

    @tf.function(autograph=True)
    def _valid_one_step(self, inputs: tf.Tensor, target: tf.Tensor, **kwargs) -> tf.Tensor:
        loss = 0
        enc_output, states = self.encoder(inputs=inputs)
        dec_input = tf.expand_dims(input=[kwargs.get("start_sign", 2)] * self.batch_size, axis=1)
        for t in range(1, self.max_sentence):
            predictions, states, _ = self.decoder(inputs=[dec_input, enc_output, states])
            loss += loss_func_mask(real=target[:, t], pred=predictions)
            dec_input = tf.expand_dims(target[:, t], 1)

            self.accuracy_metric(target[:, t], predictions)

        return loss

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
        enc_output, states = self.encoder(inputs=enc_input)
        dec_input = tf.expand_dims([tokenizer.word_index.get(start_sign)], 0)

        beam_search_container = BeamSearch(beam_size=beam_size, max_length=self.max_sentence, worst_score=0)
        beam_search_container.reset(enc_output=enc_output, dec_input=dec_input, remain=states)
        enc_output, dec_input, states = beam_search_container.get_search_inputs()

        for t in range(self.max_sentence):
            predictions, _, _ = self.decoder(inputs=[dec_input, enc_output, states])
            predictions = tf.nn.softmax(predictions, axis=-1)

            beam_search_container.expand(predictions=predictions, end_sign=tokenizer.word_index.get(end_sign))
            if beam_search_container.beam_size == 0:
                break

            enc_output, dec_input, states = beam_search_container.get_search_inputs()
            dec_input = tf.expand_dims(input=dec_input[:, -1], axis=-1)

        beam_search_result = beam_search_container.get_result(top_k=3)
        result = ""
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = tokenizer.sequences_to_texts(temp)
            text[0] = text[0].replace(start_sign, "").replace(end_sign, "").replace(" ", "")
            result = "<" + text[0] + ">" + result
        return result
