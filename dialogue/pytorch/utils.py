import os
import copy
import torch
from optparse import OptionParser


class Tokenizer(object):
    """文本分词工具类"""
    def __init__(self):