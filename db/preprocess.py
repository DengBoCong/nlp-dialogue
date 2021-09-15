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
from dialogue.tokenizer import Segment


class DataBase(abc.ABC):
    """ 数据格式基类
    """
    pass


# 在这里面进行向量化npy

class Qa(object):
    """ QA pair形式数据类型
    """

    def __init__(self, segment: Segment = None):
        """ 后面不会用到分词的话就不用传segment
        :param segment: 分词器
        :return: None
        """

    def to_npy(self, tokens_list: list = None, file_path: str = None, file_list: list = None, split: str = None):
        """ 保存为单个npy文件
        :param tokens_list: 未分词或已分词的文本列表或token列表
        :param file_path: 未分词或已分词文本列表文件路径，一行一个文本
        :param file_list: 未分词或已分词的文本列表文件路径列表，一行一个文本
        :param split: 文本分隔符，list模式不传则每个element视为list，file模式必传
        :return: None
        """
        pass
    
