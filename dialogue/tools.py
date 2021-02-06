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
"""全局公用工具模块
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import json
import time
import jieba
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from collections import OrderedDict


def log_operator(level: str, log_file: str = None,
                 log_format: str = "[%(levelname)s] - [%(asctime)s] - [file: %(filename)s] - "
                                   "[function: %(funcName)s] - [%(message)s]") -> logging.Logger:
    """ 日志操作方法，日志级别有"CRITICAL","FATAL","ERROR","WARN","WARNING","INFO","DEBUG","NOTSET"
    CRITICAL = 50, FATAL = CRITICAL, ERROR = 40, WARNING = 30, WARN = WARNING, INFO = 20, DEBUG = 10, NOTSET = 0

    :param log_file: 日志路径
    :param level: 日志级别
    :param log_format: 日志信息格式
    :return: 日志记录器
    """
    if log_file is None:
        log_file = os.path.abspath(__file__)[
                   :os.path.abspath(__file__).rfind("\\dialogue\\")] + "\\dialogue\\data\\preprocess\\runtime.log"

    logger = logging.getLogger()
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level=level)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def show_history(history: dict, save_dir: str, valid_freq: int):
    """ 用于显示历史指标趋势以及保存历史指标图表图

    :param history: 历史指标
    :param save_dir: 历史指标显示图片保存位置
    :param valid_freq: 验证频率
    :return: 无返回值
    """
    train_x_axis = [i + 1 for i in range(len(history["train_loss"]))]
    valid_x_axis = [(i + 1) * valid_freq for i in range(len(history["valid_loss"]))]

    figure, axis = plt.subplots(1, 1)
    tick_spacing = 1
    if len(history["train_loss"]) > 20:
        tick_spacing = len(history["train_loss"]) // 20
    plt.plot(train_x_axis, history["train_loss"], label="train_loss", marker=".")
    plt.plot(train_x_axis, history["train_accuracy"], label="train_accuracy", marker=".")
    plt.plot(valid_x_axis, history["valid_loss"], label="valid_loss", marker=".", linestyle="--")
    plt.plot(valid_x_axis, history["valid_accuracy"], label="valid_accuracy", marker=".", linestyle="--")
    plt.xticks(valid_x_axis)
    plt.xlabel("epoch")
    plt.legend()

    axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    save_path = save_dir + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.show()


class ProgressBar(object):
    """ 进度条工具 """

    EXECUTE = "%(current)d/%(total)d %(bar)s (%(percent)3d%%) %(metrics)s"
    DONE = "%(current)d/%(total)d %(bar)s - %(time).4fs/step %(metrics)s"

    def __init__(self, total: int = 100, num: int = 1, width: int = 30, fmt: str = EXECUTE,
                 symbol: str = "=", remain: str = ".", output=sys.stderr):
        """
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        assert len(symbol) == 1
        self.args = {}
        self.metrics = ""
        self.total = total
        self.num = num
        self.width = width
        self.symbol = symbol
        self.remain = remain
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

    def __call__(self, current: int, metrics: str):
        """
        :param current: 已执行次数
        :param metrics: 附加在进度条后的指标字符串
        """
        self.metrics = metrics
        percent = current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + ">" + self.remain * (self.width - size - 1) + "]"

        self.args = {
            "total": self.total * self.num,
            "bar": bar,
            "current": current * self.num,
            "percent": percent * 100,
            "metrics": metrics
        }
        print("\r" + self.fmt % self.args, file=self.output, end="")

    def reset(self, total: int, num: int, width: int = 30, fmt: str = EXECUTE,
              symbol: str = "=", remain: str = ".", output=sys.stderr):
        """重置内部属性

        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        self.__init__(total=total, num=num, width=width, fmt=fmt,
                      symbol=symbol, remain=remain, output=output)

    def done(self, step_time: float, fmt=DONE):
        """
        :param step_time: 该时间步执行完所用时间
        :param fmt: 执行完成之后进度条格式
        """
        self.args["bar"] = "[" + self.symbol * self.width + "]"
        self.args["time"] = step_time
        print("\r" + fmt % self.args + "\n", file=self.output, end="")


def get_dict_string(data: dict, prefix: str = "- ", precision: str = ": {:.4f} "):
    """将字典数据转换成key——value字符串

    :param data: 字典数据
    :param prefix: 组合前缀
    :param precision: key——value打印精度
    :return: 字符串
    """
    result = ""
    for key, value in data.items():
        result += (prefix + key + precision).format(value)

    return result


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """ 填充序列，如果未指定最大长度，则默认使用序列中最长长度

    sequences: 需要填充的序列
    maxlen: 最大长度
    dtype: 输出类型
    padding: 填充类型，pre在前，post在后
    truncating: 截断类型，pre在前，post在后
    value: 填充值类型，float或者是string
    :return: 形状为(len(sequences), maxlen)的numpy数组
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
    """ 将文本转化为token序列

    :param text: 文本列表
    :param filters: 过滤规则，默认过滤所有标点符号、制表符、换行符等
    :param lower: 是否将文本转化为lowercase
    :param split: 分隔符
    """
    if lower:
        text = text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


class Tokenizer(object):
    """文本分词工具类"""

    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                 split=' ', char_level=False, oov_token=None, document_count=0) -> None:
        """
        :param num_words: 最大token保存数量，基于出现频率
        :param filters: 过滤规则，默认过滤所有标点符号、制表符、换行符等
        :param lower: 是否将文本转化为lowercase
        :param split: 分隔符
        :param char_level: 是否以字符为token
        :param oov_token: 未登录词
        :param document_count: 总的文本句子数量
        """

        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts: list) -> None:
        """ 更新内部词汇表

        :param texts: 文本列表
        :return: 转换后的序列
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # 统计一个token出现在多少个文本中
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # 将未登录词放在词汇表首位
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # 索引0作为保留索引
        self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts) -> list:
        """ 将文本序列转化为token序列，注意了，只有前
        num_words个token才会被转换，其余转换为token词

        :param texts: 文本列表
        :return: 转换后的序列
        """
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        """ 将文本序列转化为token序列的生成器
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def sequences_to_texts(self, sequences) -> list:
        """ 将token序列转化为文本序列的生成器

        :param sequences: token序列
        :return: 转换后的文本序列
        """
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        """ 将token序列转化为文本序列，注意了，只有前
        num_words个token才会被转换，其余转换为token词

        :param sequences: token序列
        :return: 转换后的文本序列
        """
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = ' '.join(vect)
            yield vect

    def get_config(self) -> dict:
        """ 获取分词器的配置字典 """
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)

        return {
            'num_words': self.num_words,
            'filters': self.filters,
            'lower': self.lower,
            'split': self.split,
            'char_level': self.char_level,
            'oov_token': self.oov_token,
            'document_count': self.document_count,
            'word_counts': json_word_counts,
            'word_docs': json_word_docs,
            'index_docs': json_index_docs,
            'index_word': json_index_word,
            'word_index': json_word_index
        }

    def to_json(self, **kwargs) -> str:
        """ 将分词器相关数据转化为json格式返回
        """
        config = self.get_config()
        tokenizer_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(tokenizer_config, **kwargs)


def tokenizer_from_json(json_string) -> Tokenizer:
    """ 将Tokenizer序列化的json转化为Tokenizer实例

    :param json_string: json字符串
    :return: 分词器
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer


def load_tokenizer(dict_path: str) -> Tokenizer:
    """ 通过字典加载tokenizer

    :param dict_path: 字典路径
    :return tokenizer: 分词器
    """
    if not os.path.exists(dict_path):
        print("字典不存在，请检查之后重试")
        exit(0)

    with open(dict_path, "r", encoding="utf-8") as dict_file:
        json_string = dict_file.read().strip().strip("\n")
        tokenizer = tokenizer_from_json(json_string)

    return tokenizer


def preprocess_request(sentence: str, max_length: int, tokenizer: Tokenizer,
                       start_sign: str = "<start>", end_sign: str = "<end>"):
    """ 用于处理回复功能的输入句子，返回模型使用的序列

    :param sentence: 待处理句子
    :param max_length: 单个句子最大长度
    :param tokenizer: 分词器
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :return: 处理好的句子和decoder输入
    """
    sentence = " ".join(jieba.cut(sentence))
    sentence = start_sign + " " + sentence + " " + end_sign

    inputs = tokenizer.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs, maxlen=max_length, padding="post")

    return inputs
