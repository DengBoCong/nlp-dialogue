import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.utils.optimizers as optimizers
import hlp.chat.common.pre_treat as pre_treat
import hlp.chat.model.transformer as transformer
from hlp.chat.chatter import Chatter
from hlp.chat.common.utils import log_operator


class TransformerChatter(Chatter):
    """
    Transformer模型的聊天类
    """

    def __init__(self, execute_type: str, checkpoint_dir: str, num_layers: int,
                 units: int, d_model: int, num_heads: int, dropout: float, start_sign: str,
                 end_sign: str, beam_size: int, vocab_size: int, dict_fn: str, max_length: int):
        """
        Transformer聊天器初始化，用于加载模型
        :param execute_type: 对话执行模式
        :param checkpoint_dir: 检查点保存目录路径
        :param num_layers: transformer内部层数
        :param units: 单元数
        :param d_model: 嵌入层维度
        :param num_heads: 注意力头数
        :param dropout: 采样率
        :param start_sign: 开始标记
        :param end_sign: 结束标记
        :param beam_size: batch大小
        :param vocab_size: 词汇量大小
        :param dict_fn: 保存字典路径
        :param max_length: 单个句子最大长度
        :return: 无返回值
        """
        super().__init__(checkpoint_dir, beam_size, max_length, dict_fn, start_sign, end_sign)

        self.model = transformer.transformer(vocab_size=vocab_size, num_layers=num_layers, units=units,
                                             d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.learning_rate = optimizers.CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.checkpoint = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)

        print('正在检查是否存在检查点')
        if self.ckpt:
            print('存在检查点，正在加载检查点')
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

        log_operator(level=10).info("启动Transformer聊天器，执行类别为：{}，模型参数配置为：num_layers：{}，"
                                    "d_model：{}，num_heads：{}，units：{}，dropout：{}，vocab_size：{}，"
                                    "max_length：{}".format(execute_type, num_layers, d_model, num_heads,
                                                           units, dropout, vocab_size, max_length))

    def _init_loss_accuracy(self):
        """
        重置损失和精度
        """
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

    def _train_step(self, inp: tf.Tensor, tar: tf.Tensor, weight: tf.Tensor = None):
        """
        :param inp: 输入序列
        :param tar: 目标序列
        :param weight: 样本权重序列
        :return: 返回训练损失和精度
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self.model(inputs=[inp, tar_inp])
            loss = optimizers.loss_func_mask(tar_real, predictions, weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

        return self.train_loss.result(), self.train_accuracy.result()

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        获取目前已经保存在容器中的序列
        :param inputs: 对话中的问句
        :param dec_input: 对话中的答句
        :param t: 记录时间步
        :return: predictions预测
        """
        predictions = self.model(inputs=[inputs, dec_input], training=False)
        predictions = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions[:, -1:, :]
        predictions = tf.squeeze(predictions, axis=1)
        return predictions


def main():
    parser = ArgumentParser(description='%transformer chatbot V1.2.1')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--num_layers', default=2, type=int, required=False, help='encoder和decoder的内部层数')
    parser.add_argument('--d_model', default=256, type=int, required=False, help='特征维深度')
    parser.add_argument('--num_heads', default=8, type=int, required=False, help='头注意力数量')
    parser.add_argument('--units', default=512, type=int, required=False, help='隐藏层单元数')
    parser.add_argument('--dropout', default=0.1, type=float, required=False, help='dropout')
    parser.add_argument('--vocab_size', default=1500, type=int, required=False, help='词汇大小')
    parser.add_argument('--embedding_dim', default=256, type=int, required=False, help='嵌入层维度大小')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='用于训练的最大数据大小')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='用于验证的最大数据大小')
    parser.add_argument('--max_length', default=40, type=int, required=False, help='单个序列的最大长度')
    parser.add_argument('--dict_file', default='\\data\\transformer_dict.json', type=str, required=False, help='字典路径')
    parser.add_argument('--checkpoint', default='\\checkpoints\\transformer', type=str, required=False, help='检查点路径')
    parser.add_argument('--resource_data', default='\\data\\LCCC.json', type=str, required=False, help='原始数据集路径')
    parser.add_argument('--tokenized_data', default='\\data\\lccc_tokenized.txt', type=str, required=False,
                        help='处理好的多轮分词数据集路径')
    parser.add_argument('--qa_tokenized_data', default='\\data\\tokenized.txt', type=str, required=False,
                        help='处理好的单轮分词数据集路径')
    parser.add_argument('--history_image_dir', default='\\data\\history\\transformer\\', type=str, required=False,
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
    parser.add_argument('--start_sign', default='<start>', type=str, required=False, help='序列开始标记')
    parser.add_argument('--end_sign', default='<end>', type=str, required=False, help='序列结束标记')
    parser.add_argument('--unk_sign', default='<unk>', type=str, required=False, help='未登录词')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以chat目录下为基准配置
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\transformer")]
    execute_type = options['act']

    if execute_type == 'train':
        chatter = TransformerChatter(
            execute_type=execute_type, checkpoint_dir=work_path + options['checkpoint'],
            num_layers=options['num_layers'], units=options['units'], d_model=options['d_model'],
            num_heads=options['num_heads'], dropout=options['dropout'], beam_size=options['beam_size'],
            start_sign=options['start_sign'], end_sign=options['end_sign'], vocab_size=options['vocab_size'],
            dict_fn=work_path + options['dict_file'], max_length=options['max_length'])

        chatter.train(
            chatter.checkpoint, valid_data_fn='', data_fn=work_path + options['qa_tokenized_data'],
            batch_size=options['batch_size'], buffer_size=options['buffer_size'], epochs=options['epochs'],
            valid_freq=options['valid_freq'], max_valid_data_size=options['max_valid_data_size'],
            max_train_data_size=options['max_train_data_size'], valid_data_split=options['valid_data_split'],
            checkpoint_save_freq=options['checkpoint_save_freq'], checkpoint_save_size=options['checkpoint_save_size'],
            save_dir=work_path + options['history_image_dir'])

    elif execute_type == 'chat':
        chatter = TransformerChatter(
            execute_type=execute_type, checkpoint_dir=work_path + options['checkpoint'],
            num_layers=options['num_layers'], units=options['units'], d_model=options['d_model'],
            num_heads=options['num_heads'], dropout=options['dropout'], beam_size=options['beam_size'],
            start_sign=options['start_sign'], end_sign=options['end_sign'], vocab_size=options['vocab_size'],
            dict_fn=work_path + options['dict_file'], max_length=options['max_length'])

        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_datasets(
            dataset_name="lccc", raw_data_path=work_path + options['resource_data'],
            tokenized_data_path=work_path + options['tokenized_data'], remove_tokenized=True)
        pre_treat.to_single_turn_dataset(
            tokenized_data_path=work_path + options['tokenized_data'], dict_path=work_path + options['dict_file'],
            unk_sign=options['unk_sign'], start_sign=options['start_sign'], end_sign=options['end_sign'],
            max_data_size=options['max_train_data_size'], vocab_size=options['vocab_size'],
            qa_data_path=work_path + options['qa_tokenized_data'])
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    Transformer入口：指令需要附带运行参数
    cmd：python transformer_chatter.py --act [执行模式]
    执行类别：pre_treat/train/chat，默认为pre_treat
    其他参数参见main方法

    chat模式下运行时，输入ESC即退出对话
    """
    main()
