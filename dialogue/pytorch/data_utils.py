import os
import json
import jieba
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchnlp.encoders.text import pad_tensor, stack_and_pad_tensors, StaticTokenizerEncoder





def preprocess_sentence(start_sign, end_sign, sentence):
    """
    用于给句子首尾添加start和end
    Args:
        start_sign: 开始标记
        end_sign: 结束标记
        sentence: 处理的语句
    Returns:
        合成之后的句子
    """
    sentence = start_sign + ' ' + sentence + ' ' + end_sign
    return sentence


def preprocess_request(sentence, start_sign, end_sign, token, max_length):
    sentence = " ".join(jieba.cut(sentence))
    sentence = preprocess_sentence(start_sign, end_sign, sentence)
    inputs = [token.get(i, 3) for i in sentence.split(' ')]
    inputs = torch.tensor(inputs)
    inputs = [pad_tensor(tensor=inputs[:max_length], length=max_length, padding_index=0)]
    inputs = stack_and_pad_tensors(inputs)[0]
    dec_input = torch.unsqueeze(torch.tensor([token[start_sign]]), 0)

    return inputs, dec_input


def read_tokenized_data(path, start_sign, end_sign, num_examples):
    """
    用于将分词文本读入内存，并整理成问答对，返回的是整理好的文本问答对以及权重
    Args:
        path: 分词文本路径
        start_sign: 开始标记
        end_sign: 结束标记
        num_examples: 读取的数据量大小
    Returns:
        zip(*word_pairs): 整理好的问答对
        diag_weight: 样本权重
    """
    if not os.path.exists(path):
        print('不存在已经分词好的文件，请先执行pre_treat模式')
        exit(0)

    with open(path, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split('\n')
        diag_weight = []
        word_pairs = []

        # 这里如果num_examples为0的话，则读取全部文本数据，不为0则读取指定数量数据
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            # 文本数据中的问答对权重通过在问答对尾部添加“<|>”配置
            temp = line.split("<|>")
            word_pairs.append([preprocess_sentence(start_sign, end_sign, sentence) for sentence in temp[0].split('\t')])
            # 如果没有配置对应问答对权重，则默认为1.
            if len(temp) == 1:
                diag_weight.append(1.)
            else:
                diag_weight.append(float(temp[1]))

    return zip(*word_pairs), diag_weight


def load_token_dict(dict_fn):
    """
    加载字典方法
    :return:input_token, target_token
    """
    if not os.path.exists(dict_fn):
        print("不存在字典文件，请先执行train模式并生成字典文件")
        exit(0)

    with open(dict_fn, 'r', encoding='utf-8') as file:
        token = json.load(file)

    return token


class PairDataset(Dataset):
    """
    专门用于问答对形式的数据集构建的dataset，用于配合DataLoader使用
    """

    def __init__(self, input, target, diag_weight):
        self.input_tensor = input
        self.target_tensor = target
        self.diag_weight = diag_weight

    def __getitem__(self, item):
        return self.input_tensor[item], self.target_tensor[item], self.diag_weight[item]

    def __len__(self):
        return len(self.input_tensor)


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    result = []
    for text in sequences:
        temp = ''
        for token in text:
            temp = temp + ' ' + token_dict[str(token)]
        result.append(temp)
    return result
