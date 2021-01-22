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
"""seq2seq结构的实现执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser


def tf_seq2seq() -> None:
    parser = ArgumentParser(description="seq2seq chatbot")
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--units', default=1024, type=int, required=False, help='隐藏层单元数')
    parser.add_argument('--vocab_size', default=1000, type=int, required=False, help='词汇大小')
    parser.add_argument('--embedding_dim', default=256, type=int, required=False, help='嵌入层维度大小')
    parser.add_argument('--encoder_layers', default=2, type=int, required=False, help='encoder的层数')
    parser.add_argument('--decoder_layers', default=2, type=int, required=False, help='decoder的层数')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='用于训练的最大数据大小')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='用于验证的最大数据大小')
    parser.add_argument('--max_length', default=40, type=int, required=False, help='单个序列的最大长度')
    parser.add_argument('--dict_file', default='\\data\\seq2seq_dict.json', type=str, required=False, help='字典路径')
    parser.add_argument('--checkpoint', default='\\checkpoints\\seq2seq', type=str, required=False, help='检查点路径')
    parser.add_argument('--resource_data', default='\\data\\LCCC.json', type=str, required=False, help='原始数据集路径')
    parser.add_argument('--tokenized_data', default='\\data\\lccc_tokenized.txt', type=str, required=False,
                        help='处理好的多轮分词数据集路径')
    parser.add_argument('--qa_tokenized_data', default='\\data\\tokenized.txt', type=str, required=False,
                        help='处理好的单轮分词数据集路径')
    parser.add_argument('--history_image_dir', default='\\data\\history\\seq2seq\\', type=str, required=False,
                        help='数据指标图表保存路径')
    parser.add_argument('--valid_data_file', default='', type=str, required=False, help='验证数据集路径')
    parser.add_argument('--valid_freq', default=5, type=int, required=False, help='验证频率')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--checkpoint_save_size', default=1, type=int, required=False, help='单轮训练中检查点保存数量')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='Dataset加载缓冲大小')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='BeamSearch的beam大小')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据集中划分验证数据的比例')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练步数')
    parser.add_argument('--start_sign', default='start', type=str, required=False, help='序列开始标记')
    parser.add_argument('--end_sign', default='end', type=str, required=False, help='序列结束标记')
    parser.add_argument('--unk_sign', default='<unk>', type=str, required=False, help='未登录词')


if __name__ == '__main__':
    tf_seq2seq()
