import json
import os
import sys
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.chat.common.pre_treat as pre_treat
import hlp.chat.model.seq2seq as seq2seq
from hlp.chat.chatter import Chatter
from hlp.chat.common.utils import log_operator


class Seq2SeqChatter(Chatter):
    """
    Seq2Seq模型的聊天类
    """

    def __init__(self, execute_type: str, checkpoint_dir: str, units: int, embedding_dim: int, batch_size: int,
                 start_sign: str, end_sign: str, beam_size: int, vocab_size: int, dict_fn: str, max_length: int,
                 encoder_layers: int, decoder_layers: int, cell_type: str, if_bidirectional: bool = True):
        """
        Seq2Seq聊天器初始化，用于加载模型
        :param execute_type: 对话执行模式
        :param checkpoint_dir: 检查点保存目录路径
        :param units: 单元数
        :param embedding_dim: 嵌入层维度
        :param batch_size: batch大小
        :param start_sign: 开始标记
        :param end_sign: 结束标记
        :param beam_size: batch大小
        :param vocab_size: 词汇量大小
        :param dict_fn: 保存字典路径
        :param max_length: 单个句子最大长度
        :param encoder_layers: encoder中内部RNN层数
        :param decoder_layers: decoder中内部RNN层数
        :param cell_type: cell类型，lstm/gru， 默认lstm
        :param if_bidirectional: 是否双向
        :return: 无返回值
        """
        super().__init__(checkpoint_dir, beam_size, max_length, dict_fn, start_sign, end_sign)
        self.units = units
        self.batch_size = batch_size
        self.enc_units = units

        self.encoder = seq2seq.encoder(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                       enc_units=int(units / 2), num_layers=encoder_layers,
                                       cell_type=cell_type, if_bidirectional=if_bidirectional)
        self.decoder = seq2seq.decoder(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                       enc_units=units, dec_units=units,
                                       num_layers=decoder_layers, cell_type=cell_type)

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)

        print('正在检查是否存在检查点')
        if self.ckpt:
            print('存在检查点，正在加载检查点')
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，从头开始训练')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

        log_operator(level=10).info("启动SMN聊天器，执行类别为：{}，模型参数配置为：vocab_size：{}，"
                                    "embedding_dim：{}，units：{}，max_length：{}".format(execute_type, vocab_size,
                                                                                     embedding_dim, units,
                                                                                     max_length))

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
        :return: 每步损失和精度
        """
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inputs=inp)
            dec_hidden = enc_hidden
            # 这里初始化decoder的输入，首个token为start
            dec_input = tf.expand_dims([2] * self.batch_size, 1)
            for t in range(1, tar.shape[1]):
                predictions, dec_hidden, attention_weight = self.decoder(inputs=[dec_input, enc_output, dec_hidden])
                loss += self._loss_function(tar[:, t], predictions, weight)

                if sum(tar[:, t]) == 0:
                    break
                dec_input = tf.expand_dims(tar[:, t], 1)
                self.train_accuracy(tar[:, t], predictions)

        self.train_loss(loss)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return self.train_loss.result(), self.train_accuracy.result()

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        获取目前已经保存在容器中的序列
        :param inputs: 对话中的问句
        :param dec_input: 对话中的答句
        :param t: 记录时间步
        :return: predictions预测
        """
        hidden = tf.zeros((inputs.shape[0], self.units))
        enc_output, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(dec_input[:, t], 1)
        predictions, _, _ = self.decoder(inputs=[dec_input, enc_output, dec_hidden])
        return predictions

    def _loss_function(self, real: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor = None):
        """
        用于计算预测损失，注意要将填充的0进行mask，不纳入损失计算
        :param real: 真实序列
        :param pred: 预测序列
        :param weights: 样本数据的权重
        :return: 该batch的平均损失
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        if weights is not None:
            loss_ = self.loss_object(real, pred, sample_weight=weights)
        else:
            loss_ = self.loss_object(real, pred)
        # 这里要注意了，因为前面我们对于短的句子进行了填充，所
        # 以对于填充的部分，我们不能用于计算损失，所以要mask
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


def main():
    parser = ArgumentParser(description='%seq2seq chatbot V1.2.1')
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

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以chat目录下为基准配置
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\seq2seq")]
    execute_type = options['act']

    if execute_type == 'train':
        print("开始训练模型...")
        chatter = Seq2SeqChatter(
            execute_type=execute_type, checkpoint_dir=work_path + options['checkpoint'], units=options['units'],
            beam_size=options['beam_size'], embedding_dim=options['embedding_dim'], batch_size=options['batch_size'],
            start_sign=options['start_sign'], end_sign=options['end_sign'], vocab_size=options['vocab_size'],
            dict_fn=work_path + options['dict_file'], max_length=options['max_length'], if_bidirectional=True,
            encoder_layers=options['encoder_layers'], decoder_layers=options['decoder_layers'], cell_type='lstm')
        chatter.train(chatter.checkpoint, valid_data_fn='', data_fn=work_path + options['qa_tokenized_data'],
                      valid_data_split=options['valid_data_split'], save_dir=work_path + options['history_image_dir'],
                      max_valid_data_size=options['max_valid_data_size'], valid_freq=options['valid_freq'],
                      max_train_data_size=options['max_train_data_size'], epochs=options['epochs'],
                      checkpoint_save_freq=options['checkpoint_save_freq'], batch_size=options['batch_size'],
                      checkpoint_save_size=options['checkpoint_save_size'], buffer_size=options['buffer_size'])
    elif execute_type == 'chat':
        chatter = Seq2SeqChatter(
            execute_type=execute_type, checkpoint_dir=work_path + options['checkpoint'], units=options['units'],
            beam_size=options['beam_size'], embedding_dim=options['embedding_dim'], batch_size=options['batch_size'],
            start_sign=options['start_sign'], end_sign=options['end_sign'], vocab_size=options['vocab_size'],
            dict_fn=work_path + options['dict_file'], max_length=options['max_length'], if_bidirectional=True,
            encoder_layers=options['encoder_layers'], decoder_layers=options['decoder_layers'], cell_type='lstm')
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_datasets(dataset_name="lccc", raw_data_path=work_path + options['resource_data'],
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
    Seq2Seq入口：指令需要附带运行参数
    cmd：python seq2seq2_chatter.py --type [执行模式]
    执行类别：pre_treat/train/chat，默认为pre_treat
    其他参数参见main方法

    chat模式下运行时，输入ESC即退出对话
    """
    main()
