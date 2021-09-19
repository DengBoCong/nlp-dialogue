#! -*- coding: utf-8 -*-
""" 数据预料处理
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
from dialogue.tokenizer import pad_sequences
from dialogue.tokenizer import Segment
from dialogue.tokenizer import Tokenizer


class DataProcessor(abc.ABC):
    """ 数据格式基类
    """

    def __init__(self, tokenizer: Tokenizer, segment: Segment = None):
        """
        :param tokenizer: token处理器，这个必传
        :param segment: 分词器，后面不会用到分词的话就不用传segment
        :return: None
        """
        self.tokenizer = tokenizer
        self.segment = segment

    @abc.abstractmethod
    def to_npy(self, *args, **kwargs):
        """ 该方法用于将处理好的语料数据转换成npy文件格式

        Note:
            a):
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def to_file(self, *args, **kwargs):
        """ 该方法用于将处理好的语料数据转换成文件格式

        Note:
            a):
        """


class TextPair(DataProcessor):
    """ text pair或text pair + label形式数据类型
    """

    def __init__(self, tokenizer: Tokenizer, segment: Segment = None):
        """
        :param tokenizer: token处理器，这个必传
        :param segment: 分词器，后面不会用到分词的话就不用传segment
        :return: None
        """
        super(TextPair, self).__init__(tokenizer, segment)

    def to_npy(self, batch_size: int, output_dir: str, file_path: str,
               split: str, if_seg: bool = False, d_type: str = "int32") -> None:
        """ 保存为npy文件
        :param batch_size: 每个npy文件保存的样本数
        :param output_dir: 文件输出目录
        :param file_path: 未分词或已分词文本列表文件路径，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        :param if_seg: 是否进行分词，注意使用需要初始化传入segment
        :param d_type: label数据类型
        :return: None
        """
        if if_seg and not self.segment:
            raise TypeError("Segment must be instantiated in the init method")

        with open(os.path.join(output_dir, "outputs.txt"), "w", encoding="utf-8"
                  ) as output_file, open(file_path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip().strip("\n")
                if line == "":
                    continue

                elements = line.split(split)
                if len(elements) < 2 or len(elements) > 3:
                    raise RuntimeError("TextPair - to_npy: The data does not meet the format requirements")


if __name__ == "__main__":
    print([1, 2, "fsd", 1.2])
