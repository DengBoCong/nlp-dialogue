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
"""模型评估指标
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from typing import List


def recall_at_position_k_in_n(labels: list, k: list = [1], n: int = 10, tar: float = 1.0) -> List:
    """ Rn@k 召回率指标计算

    :param labels: 数据列表
    :param k: top k
    :param n: 样本范围
    :param tar: 目标值
    :return: 所得指标值
    """
    score = labels[0]
    label = labels[1]

    length = len(k)
    sum_k = [0.0] * length
    total = 0
    for i in range(0, len(label), n):
        total += 1
        remain = [label[index] for index in np.argsort(score[i:i + n])]
        for j in range(length):
            sum_k[j] += 1.0 * remain[-k[j]:].count(tar) / remain.count(tar)

    for i in range(length):
        sum_k[i] /= total

    return sum_k
