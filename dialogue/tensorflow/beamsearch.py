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
"""BeachSearch实现，支持Encoder-Decoder结构，多beam输出和截断
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf


class BeamSearch(object):

    def __init__(self, beam_size, max_length, worst_score):
        """
        :param beam_size: beam大小
        :param max_length: 句子最大长度
        :param worst_score: 最差分数
        """
        self.BEAM_SIZE = beam_size  # 保存原始beam大小，用于重置
        self.MAX_LEN = max_length - 1
        self.MIN_SCORE = worst_score  # 保留原始worst_score，用于重置

        self.candidates = []  # 保存中间状态序列的容器，元素格式为(score, sequence)类型为(float, [])
        self.result = []  # 用来保存已经遇到结束符的序列
        self.result_plus = []  # 用来保存已经遇到结束符的带概率分布的序列
        self.candidates_plus = []  # 保存已经遇到结束符的序列及概率分布

        self.enc_output = None
        self.remain = None
        self.dec_inputs = None
        self.beam_size = None
        self.worst_score = None

    def __len__(self):
        """当前候选结果数
        """
        return len(self.candidates)

    def reset(self, enc_output: tf.Tensor, dec_input: tf.Tensor, remain: tf.Tensor):
        """重置搜索

        :param enc_output: 已经序列化的输入句子
        :param dec_input: 解码器输入序列
        :param remain: 预留decoder输入
        :return: 无返回值
        """
        self.candidates = []  # 保存中间状态序列的容器，元素格式为(score, sequence)类型为(float, [])
        self.candidates_plus = []  # 保存已经遇到结束符的序列及概率分布,元素为(score, tensor),tensor的shape为(seq_len, vocab_size)
        self.candidates.append((1, dec_input))
        self.enc_output = enc_output
        self.remain = remain
        self.dec_inputs = dec_input
        self.beam_size = self.BEAM_SIZE  # 新一轮中，将beam_size重置为原beam大小
        self.worst_score = self.MIN_SCORE  # 新一轮中，worst_score重置
        self.result = []  # 用来保存已经遇到结束符的序列
        self.result_plus = []  # 用来保存已经遇到结束符的带概率分布的序列元素为tensor, tensor的shape为(seq_len, vocab_size)

    def get_search_inputs(self):
        """为下一步预测生成输入

        :return: enc_output, dec_inputs, remain
        """
        # 生成多beam输入
        enc_output = self.enc_output
        remain = self.remain
        temp = self.candidates[0][1]
        for i in range(1, len(self)):
            enc_output = tf.concat([enc_output, self.enc_output], 0)
            remain = tf.concat([remain, self.remain], 0)
            temp = tf.concat([temp, self.candidates[i][1]], axis=0)
        self.dec_inputs = copy.deepcopy(temp)

        return enc_output, self.dec_inputs, remain

    def _reduce_end(self, end_sign):
        """ 当序列遇到了结束token，需要将该序列从容器中移除

        :param end_sign: 句子结束标记
        :return: 无返回值
        """
        for idx, (s, dec) in enumerate(self.candidates):
            temp = dec.numpy()
            if temp[0][-1] == end_sign:
                self.result.append((self.candidates[idx][0], self.candidates[idx][1]))
                self.result_plus.append(self.candidates_plus[idx])
                del self.candidates[idx]
                del self.candidates_plus[idx]
                self.beam_size -= 1

    def expand(self, predictions, end_sign):
        """ 根据预测结果对候选进行扩展
        往容器中添加预测结果，在本方法中对预测结果进行整理、排序的操作

        :param predictions: 传入每个时间步的模型预测值
        :param end_sign: 句子结束标记
        :return: 无返回值
        """
        prev_candidates = copy.deepcopy(self.candidates)
        prev_candidates_plus = copy.deepcopy(self.candidates_plus)
        self.candidates.clear()
        self.candidates_plus.clear()
        predictions = predictions.numpy()
        predictions_plus = copy.deepcopy(predictions)
        # 在batch_size*beam_size个prediction中找到分值最高的beam_size个
        for i in range(self.dec_inputs.shape[0]):  # 外循环遍历batch_size（batch_size的值其实就是之前选出的候选数量）
            for _ in range(self.beam_size):  # 内循环遍历选出beam_size个概率最大位置
                token_index = tf.argmax(input=predictions[i], axis=0)  # predictions.shape --> (batch_size, vocab_size)
                # 计算分数
                score = prev_candidates[i][0] * predictions[i][token_index]
                predictions[i][token_index] = 0
                # 判断容器容量以及分数比较
                if len(self) < self.beam_size or score > self.worst_score:
                    self.candidates.append(
                        (score, tf.concat([prev_candidates[i][1], tf.constant([[token_index.numpy()]], shape=(1, 1))],
                                          axis=-1)))
                    if len(prev_candidates_plus) == 0:
                        self.candidates_plus.append((score, predictions_plus))
                    else:
                        self.candidates_plus.append(
                            (score, tf.concat([prev_candidates_plus[i][1], [predictions_plus[i]]], axis=0)))
                    if len(self) > self.beam_size:
                        sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.candidates)])
                        del self.candidates[sorted_scores[0][1]]
                        del self.candidates_plus[sorted_scores[0][1]]
                        self.worst_score = sorted_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)
        self._reduce_end(end_sign=end_sign)

    def get_result(self, top_k=1):
        """获得概率最高的top_k个结果

        :param top_k: 输出结果数量
        :return: 概率最高的top_k个结果
        """
        results = [element[1] for element in sorted(self.result)[-top_k:]]
        return results

    def get_result_plus(self, top_k=1):
        """获得概率最高的top_k个结果

        :param top_k: 输出结果数量
        :return: 概率最高的top_k个带概率的结果
        """
        sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.result)], reverse=True)
        results_plus = []
        for i in range(top_k):
            results_plus.append(self.result_plus[sorted_scores[i][1]][1])

        return results_plus
