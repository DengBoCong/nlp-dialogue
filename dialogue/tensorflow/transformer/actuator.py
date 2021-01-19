import os
import json
import tensorflow as tf
from argparse import ArgumentParser
from dialogue.tensorflow.optimizers import CustomSchedule
import dialogue.tensorflow.transformer.model as transformer


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
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='用于训练的最大数据大小')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='用于验证的最大数据大小')
    parser.add_argument('--max_length', default=40, type=int, required=False, help='单个序列的最大长度')
    parser.add_argument('--valid_data_file', default='', type=str, required=False, help='验证数据集路径')
    parser.add_argument('--valid_freq', default=5, type=int, required=False, help='验证频率')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--checkpoint_save_size', default=1, type=int, required=False, help='单轮训练中检查点保存数量')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='Dataset加载缓冲大小')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='BeamSearch的beam大小')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据集中划分验证数据的比例')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练步数')
    parser.add_argument('--start_sign', default='<start>', type=str, required=False, help='序列开始标记')
    parser.add_argument('--end_sign', default='<end>', type=str, required=False, help='序列结束标记')
    parser.add_argument('--unk_sign', default='<unk>', type=str, required=False, help='未登录词')
    parser.add_argument('--dict_file', default='\\data\\transformer_dict.json', type=str, required=False, help='字典路径')
    parser.add_argument('--checkpoint', default='\\checkpoints\\transformer', type=str, required=False, help='检查点路径')
    parser.add_argument('--resource_data', default='\\data\\LCCC.json', type=str, required=False, help='原始数据集路径')
    parser.add_argument('--tokenized_data', default='\\data\\lccc_tokenized.txt', type=str, required=False,
                        help='处理好的多轮分词数据集路径')
    parser.add_argument('--qa_tokenized_data', default='\\data\\tokenized.txt', type=str, required=False,
                        help='处理好的单轮分词数据集路径')
    parser.add_argument('--history_image_dir', default='\\data\\history\\transformer\\', type=str, required=False,
                        help='数据指标图表保存路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以transformer目录下为基准配置
    file_path = os.path.abspath(__file__)
    work_path = file_path[:file_path.find("transformer")]

    encoder = transformer.encoder(vocab_size=options["vocab_size"], num_layers=options["num_layers"],
                                  units=options["units"], embedding_dim=options["embedding_dim"],
                                  num_heads=options["num_heads"], dropout=options["dropout"])
    decoder = transformer.decoder(vocab_size=options["vocab_size"], num_layers=options["num_layers"],
                                  units=options["units"], embedding_dim=options["embedding_dim"],
                                  num_heads=options["num_heads"], dropout=options["dropout"])

    learning_rate = CustomSchedule(d_model=options["embedding_dim"])
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


if __name__ == '__main__':
    main()
