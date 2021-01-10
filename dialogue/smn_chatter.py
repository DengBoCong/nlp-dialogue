import os
import sys
import json
import time
import pysolr
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.chat.model.smn as smn
import hlp.chat.common.utils as utils
import hlp.chat.common.data_utils as data_utils
from hlp.utils.utils import load_tokenizer


class SMNChatter():
    """
    SMN的聊天器
    """

    def __init__(self, units: int, vocab_size: int, execute_type: str, dict_fn: str,
                 embedding_dim: int, checkpoint_dir: int, max_utterance: int, max_sentence: int,
                 learning_rate: float, database_fn: str, solr_server: str):
        """
        SMN聊天器初始化，用于加载模型
        :param units: 单元数
        :param vocab_size: 词汇量大小
        :param execute_type: 对话执行模式
        :param dict_fn: 保存字典路径
        :param embedding_dim: 嵌入层维度
        :param checkpoint_dir: 检查点保存目录路径
        :param max_utterance: 每轮句子数量
        :param max_sentence: 单个句子最大长度
        :param learning_rate: 学习率
        :param database_fn: 候选数据库路径
        :return: 无返回值
        """
        self.dict_fn = dict_fn
        self.checkpoint_dir = checkpoint_dir
        self.max_utterance = max_utterance
        self.max_sentence = max_sentence
        self.database_fn = database_fn
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.solr = pysolr.Solr(url=solr_server, always_commit=True, timeout=10)
        self.train_loss = tf.keras.metrics.Mean()

        self.model = smn.smn(units=units, vocab_size=vocab_size,
                             embedding_dim=embedding_dim,
                             max_utterance=self.max_utterance,
                             max_sentence=self.max_sentence)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, )

        ckpt = os.path.exists(checkpoint_dir)
        if not ckpt:
            os.makedirs(checkpoint_dir)

        print('正在检查是否存在检查点')
        if ckpt:
            print('存在检查点，正在加载检查点'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

        logger = utils.log_operator(level=10)
        logger.info("启动SMN聊天器，执行类别为：{}，模型参数配置为：embedding_dim：{}，"
                    "max_sentence：{}，max_utterance：{}，units：{}，vocab_size：{}，"
                    "learning_rate：{}".format(execute_type, embedding_dim, max_sentence,
                                              max_utterance, units, vocab_size, learning_rate))

    def train(self, epochs: int, data_fn: str, batch_size: int, buffer_size: int,
              max_train_data_size: int = 0, max_valid_data_size: int = 0):
        """
        训练功能
        :param epochs: 训练执行轮数
        :param data_fn: 数据文本路径
        :param buffer_size: Dataset加载缓存大小
        :param batch_size: Dataset加载批大小
        :param max_train_data_size: 最大训练数据量
        :param max_valid_data_size: 最大验证数据量
        :return: 无返回值
        """
        # 处理并加载训练数据，
        dataset, tokenizer, checkpoint_prefix, steps_per_epoch = \
            data_utils.smn_load_train_data(dict_fn=self.dict_fn, data_fn=data_fn,
                                           buffer_size=buffer_size, batch_size=batch_size,
                                           checkpoint_dir=self.checkpoint_dir, max_utterance=self.max_utterance,
                                           max_sentence=self.max_sentence, max_train_data_size=max_train_data_size)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            self.train_loss.reset_states()

            sample_sum = 0
            batch_sum = 0

            for (batch, (utterances, response, label)) in enumerate(dataset.take(steps_per_epoch)):
                with tf.GradientTape() as tape:
                    scores = self.model(inputs=[utterances, response])
                    loss = tf.keras.losses. \
                        SparseCategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.AUTO)(label, scores)
                gradient = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
                self.train_loss(loss)

                sample_num = len(utterances)
                batch_sum += sample_num
                sample_sum = steps_per_epoch * sample_num
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum),
                      end='', flush=True)

            r2_1, _ = self.evaluate(valid_fn=data_fn,
                                    tokenizer=tokenizer,
                                    max_valid_data_size=max_valid_data_size)

            step_time = time.time() - start_time
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f} - R2@1：{:0.3f}\n'
                             .format(step_time, self.train_loss.result(), r2_1))
            sys.stdout.flush()
            self.checkpoint.save(file_prefix=checkpoint_prefix)

    def evaluate(self, valid_fn: str, dict_fn: str = "", max_turn_utterances_num: int = 10,
                 max_valid_data_size: int = 0):
        """
        验证功能，注意了dict_fn和tokenizer两个比传其中一个
        :param valid_fn: 验证数据集路径
        :param dict_fn: 字典路径
        :param max_turn_utterances_num: 最大训练数据量
        :param max_valid_data_size: 最大验证数据量
        :return: r2_1, r10_1指标
        """
        step = max_valid_data_size // max_turn_utterances_num
        if max_valid_data_size == 0:
            return None
        # 处理并加载评价数据，注意，如果max_valid_data_size传
        # 入0，就直接跳过加载评价数据，也就是说只训练不评价
        tokenizer = load_tokenizer(dict_path=dict_fn)
        valid_dataset = data_utils.load_smn_valid_data(data_fn=valid_fn,
                                                       max_sentence=self.max_sentence,
                                                       max_utterance=self.max_utterance,
                                                       tokenizer=tokenizer,
                                                       max_turn_utterances_num=max_turn_utterances_num,
                                                       max_valid_data_size=max_valid_data_size)

        scores = tf.constant([], dtype=tf.float32)
        labels = tf.constant([], dtype=tf.int32)
        for (batch, (utterances, response, label)) in enumerate(valid_dataset.take(step)):
            score = self.model(inputs=[utterances, response])
            score = tf.nn.softmax(score, axis=-1)
            labels = tf.concat([labels, label], axis=0)
            scores = tf.concat([scores, score[:, 1]], axis=0)

        r10_1 = self._metrics_rn_1(scores, labels, num=10)
        r2_1 = self._metrics_rn_1(scores, labels, num=2)
        return r2_1, r10_1

    def respond(self, req: str):
        """
        对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        :param req: 输入的语句
        :return: 系统回复字符串
        """
        self.solr.ping()
        history = req[-self.max_utterance:]
        pad_sequences = [0] * self.max_sentence
        tokenizer = load_tokenizer(self.dict_fn)
        utterance = tokenizer.texts_to_sequences(history)
        utterance_len = len(utterance)

        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != self.max_utterance:
            utterance += [pad_sequences] * (self.max_utterance - utterance_len)
        utterance = tf.keras.preprocessing.sequence.pad_sequences(utterance, maxlen=self.max_sentence,
                                                                  padding="post").tolist()

        tf_idf = data_utils.get_tf_idf_top_k(history)
        query = "{!func}sum("
        for key in tf_idf:
            query += "product(idf(utterance," + key + "),tf(utterance," + key + ")),"
        query += ")"
        candidates = self.solr.search(q=query, start=0, rows=10).docs
        candidates = [candidate['utterance'][0] for candidate in candidates]

        if candidates is None:
            return "Sorry! I didn't hear clearly, can you say it again?"
        else:
            utterances = [utterance] * len(candidates)
            responses = tokenizer.texts_to_sequences(candidates)
            responses = tf.keras.preprocessing.sequence.pad_sequences(responses, maxlen=self.max_sentence,
                                                                      padding="post")
            utterances = tf.convert_to_tensor(utterances)
            responses = tf.convert_to_tensor(responses)
            scores = self.model(inputs=[utterances, responses])
            index = tf.argmax(scores[:, 0])

            return candidates[index]

    def _metrics_rn_1(self, scores: float, labels: tf.Tensor, num: int = 10):
        """
        计算Rn@k指标
        :param scores: 训练所得分数
        :param labels: 数据标签
        :param num: n
        :return: rn_1指标
        """
        total = 0
        correct = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                total = total + 1
                sublist = scores[i:i + num]
                if max(sublist) == scores[i]:
                    correct = correct + 1
        return float(correct) / total


def main():
    parser = ArgumentParser(description='%smn multi_turn chatbot V1.2.1')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--units', default=200, type=int, required=False, help='隐藏层单元数')
    parser.add_argument('--vocab_size', default=2000, type=int, required=False, help='词汇大小')
    parser.add_argument('--embedding_dim', default=200, type=int, required=False, help='嵌入层维度大小')
    parser.add_argument('--max_sentence', default=50, type=int, required=False, help='单个句子序列最大长度')
    parser.add_argument('--max_utterance', default=10, type=int, required=False, help='当回合最大句子数')
    parser.add_argument('--max_train_data_size', default=36, type=int, required=False, help='用于训练的最大数据大小')
    parser.add_argument('--max_valid_data_size', default=100, type=int, required=False, help='用于验证的最大数据大小')
    parser.add_argument('--learning_rate', default=0.001, type=float, required=False, help='学习率')
    parser.add_argument('--max_database_size', default=0, type=int, required=False, help='最大数据候选数量')
    parser.add_argument('--dict_file', default='\\data\\smn_dict.json', type=str, required=False, help='字典路径')
    parser.add_argument('--checkpoint', default='\\checkpoints\\smn', type=str, required=False, help='检查点路径')
    parser.add_argument('--tokenized_train', default='\\data\\ubuntu_train.txt', type=str, required=False,
                        help='处理好的多轮分词训练数据集路径')
    parser.add_argument('--tokenized_valid', default='\\data\\ubuntu_valid.txt', type=str, required=False,
                        help='处理好的多轮分词验证数据集路径')
    parser.add_argument('--solr_server', default='http://49.235.33.100:8983/solr/smn/', type=str, required=False,
                        help='solr服务地址')
    parser.add_argument('--candidate_database', default='\\data\\candidate.json', type=str, required=False,
                        help='候选回复数据库')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练步数')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='Dataset加载缓冲大小')
    parser.add_argument('--unk_sign', default='<unk>', type=str, required=False, help='未登录词')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了有关路径的参数，以chat目录下为基准配置
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\smn")]
    execute_type = options['act']

    if execute_type == 'train':
        chatter = SMNChatter(
            units=options['units'], vocab_size=options['vocab_size'], execute_type=execute_type,
            dict_fn=work_path + options['dict_file'], solr_server=options['solr_server'],
            embedding_dim=options['embedding_dim'], checkpoint_dir=work_path + options['checkpoint'],
            learning_rate=options['learning_rate'], max_utterance=options['max_utterance'],
            max_sentence=options['max_sentence'], database_fn=work_path + options['candidate_database'])
        chatter.train(epochs=options['epochs'], data_fn=work_path + options['tokenized_train'],
                      batch_size=options['batch_size'], buffer_size=options['buffer_size'],
                      max_train_data_size=options['max_train_data_size'],
                      max_valid_data_size=options['max_valid_data_size'])
    elif execute_type == 'pre_treat':
        data_utils.creat_index_dataset(data_fn=work_path + options['tokenized_train'], unk_sign=options['unk_sign'],
                                       solr_sever=options['solr_server'], dict_path=work_path + options['dict_file'],
                                       max_database_size=options['max_database_size'], vocab_size=options['vocab_size'])
    elif execute_type == 'evaluate':
        chatter = SMNChatter(
            units=options['units'], vocab_size=options['vocab_size'], execute_type=execute_type,
            dict_fn=work_path + options['dict_file'], solr_server=options['solr_server'],
            embedding_dim=options['embedding_dim'], checkpoint_dir=work_path + options['checkpoint'],
            learning_rate=options['learning_rate'], max_utterance=options['max_utterance'],
            max_sentence=options['max_sentence'], database_fn=work_path + options['candidate_database'])
        r2_1, r10_1 = chatter.evaluate(valid_fn=work_path + options['tokenized_valid'],
                                       dict_fn=work_path + options['dict_file'],
                                       max_valid_data_size=options['max_valid_data_size'])
        print("指标：R2@1-{:0.3f}，R10@1-{:0.3f}".format(r2_1, r10_1))

    elif execute_type == 'chat':
        chatter = SMNChatter(
            units=options['units'], vocab_size=options['vocab_size'], execute_type=execute_type,
            dict_fn=work_path + options['dict_file'], solr_server=options['solr_server'],
            embedding_dim=options['embedding_dim'], checkpoint_dir=work_path + options['checkpoint'],
            learning_rate=options['learning_rate'], max_utterance=options['max_utterance'],
            max_sentence=options['max_sentence'], database_fn=work_path + options['candidate_database'])
        history = []  # 用于存放历史对话
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            history.append(req)
            response = chatter.respond(req=history)
            print("Agent: ", response)
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    SMN入口：指令需要附带运行参数
    cmd：python smn_chatter.py --act [执行模式]
    执行类别：pre_treat/train/evaluate/chat，默认为pre_treat
    其他参数参见main方法

    chat模式下运行时，输入ESC即退出对话
    """
    main()
