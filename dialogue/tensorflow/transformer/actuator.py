import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\dialogue\\")])
from dialogue.tensorflow.utils import load_checkpoint
from dialogue.tensorflow.optimizers import CustomSchedule
import dialogue.tensorflow.transformer.model as transformer
from dialogue.common.preprocess_corpus import preprocess_dataset
from dialogue.common.preprocess_corpus import to_single_turn_dataset


def main():
    parser = ArgumentParser(description="transformer chatbot")
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--num_layers', default=2, type=int, required=False, help='encoder和decoder的内部层数')
    parser.add_argument('--num_heads', default=8, type=int, required=False, help='头注意力数量')
    parser.add_argument('--units', default=512, type=int, required=False, help='隐藏层单元数')
    parser.add_argument('--dropout', default=0.1, type=float, required=False, help='dropout')
    parser.add_argument('--vocab_size', default=1500, type=int, required=False, help='词汇大小')
    parser.add_argument('--embedding_dim', default=256, type=int, required=False, help='嵌入层维度大小')
    parser.add_argument('--learning_rate_beta_1', default=0.9, type=int, required=False, help='一阶动量的指数加权平均权值')
    parser.add_argument('--learning_rate_beta_2', default=0.98, type=int, required=False, help='二阶动量的指数加权平均权值')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='用于训练的最大数据大小')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='用于验证的最大数据大小')
    parser.add_argument('--max_length', default=40, type=int, required=False, help='单个序列的最大长度')
    parser.add_argument('--valid_data_file', default='', type=str, required=False, help='验证数据集路径')
    parser.add_argument('--valid_freq', default=5, type=int, required=False, help='验证频率')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--checkpoint_save_size', default=3, type=int, required=False, help='单轮训练中检查点保存数量')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='Dataset加载缓冲大小')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='BeamSearch的beam大小')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据集中划分验证数据的比例')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练步数')
    parser.add_argument('--start_sign', default='<start>', type=str, required=False, help='序列开始标记')
    parser.add_argument('--end_sign', default='<end>', type=str, required=False, help='序列结束标记')
    parser.add_argument('--unk_sign', default='<unk>', type=str, required=False, help='未登录词')
    parser.add_argument('--dict_file', default='data\\transformer_dict.json', type=str, required=False, help='字典路径')
    parser.add_argument('--checkpoint_dir', default='checkpoints\\tensorflow\\transformer', type=str, required=False,
                        help='检查点路径')
    parser.add_argument('--resource_data', default='data\\LCCC.json', type=str, required=False, help='原始数据集路径')
    parser.add_argument('--tokenized_data', default='data\\preprocess\\lccc_tokenized.txt', type=str, required=False,
                        help='处理好的多轮分词数据集路径')
    parser.add_argument('--qa_tokenized_data', default='data\\preprocess\\tokenized.txt', type=str, required=False,
                        help='处理好的单轮分词数据集路径')
    parser.add_argument('--history_image_dir', default='data\\history\\transformer\\', type=str, required=False,
                        help='数据指标图表保存路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以transformer目录下为基准配置
    file_path = os.path.abspath(__file__)
    work_path = file_path[:file_path.find("tensorflow")]

    encoder = transformer.encoder(vocab_size=options["vocab_size"], num_layers=options["num_layers"],
                                  units=options["units"], embedding_dim=options["embedding_dim"],
                                  num_heads=options["num_heads"], dropout=options["dropout"])
    decoder = transformer.decoder(vocab_size=options["vocab_size"], num_layers=options["num_layers"],
                                  units=options["units"], embedding_dim=options["embedding_dim"],
                                  num_heads=options["num_heads"], dropout=options["dropout"])

    learning_rate = CustomSchedule(d_model=options["embedding_dim"])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=options["learning_rate_beta_1"],
                                   beta_2=options["learning_rate_beta_2"], epsilon=1e-9)
    checkpoint_manager = load_checkpoint(checkpoint_dir=work_path + options["checkpoint_dir"],
                                         execute_type=options["act"], encoder=encoder, decoder=decoder,
                                         checkpoint_save_size=options["checkpoint_save_size"])

    if options["act"] == "pre_treat":
        preprocess_dataset(dataset_name="lccc", raw_data_path=work_path + options['resource_data'],
                           tokenized_data_path=work_path + options['tokenized_data'], remove_tokenized=True)
        to_single_turn_dataset(tokenized_data_path=work_path + options['tokenized_data'],
                               dict_path=work_path + options['dict_file'], unk_sign=options['unk_sign'],
                               start_sign=options['start_sign'], end_sign=options['end_sign'],
                               max_data_size=options['max_train_data_size'], vocab_size=options['vocab_size'],
                               qa_data_path=work_path + options['qa_tokenized_data'])


if __name__ == '__main__':
    main()
