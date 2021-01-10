import os
import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import deque
import hlp.chat.common.data_utils as data_utils
from hlp.utils.beamsearch import BeamSearch
from hlp.utils.utils import load_tokenizer


class Chatter(object):
    """"
    面向使用者的聊天器基类
    该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
    不同模型或方法实现的聊天子类化该类。
    """

    def __init__(self, checkpoint_dir: str, beam_size: int, max_length: int, dict_fn: str, start_sign, end_sign):
        """
        聊天器初始化，用于加载模型
        :param checkpoint_dir: 检查点保存目录路径
        :param beam_size: batch大小
        :param max_length: 单个句子最大长度
        :param dict_fn: 保存字典路径
        :param start_sign: 开始标记
        :param end_sign: 结束标记
        return: 无返回值
        """
        self.max_length = max_length
        self.dict_fn = dict_fn
        self.start_sign = start_sign
        self.end_sign = end_sign
        self.checkpoint_dir = checkpoint_dir
        self.beam_search_container = BeamSearch(beam_size=beam_size, max_length=max_length, worst_score=0)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt = tf.io.gfile.listdir(checkpoint_dir)

    def _init_loss_accuracy(self):
        """
        初始化损失
        """
        pass

    def _train_step(self, inp: tf.Tensor, tar: tf.Tensor, weight: int, step_loss: float):
        """
        模型训练步方法，需要返回时间步损失
        :param inp: 输入序列
        :param tar: 目标序列
        :param weight: 样本权重序列
        :param step_loss: 每步损失
        :return: 每步损失和精度
        """
        pass

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        使用模型预测下一个Token的id
        :param inputs: 对话中的问句
        :param dec_input: 对话中的答句
        :param t: 记录时间步
        :return: predictions预测
        """
        pass

    def train(self, checkpoint: tf.train.Checkpoint, data_fn: str, batch_size: int,
              buffer_size: int, max_train_data_size: int, epochs: int, max_valid_data_size: int,
              checkpoint_save_freq: int, checkpoint_save_size: int, save_dir: str,
              valid_data_split: float = 0.0, valid_data_fn: str = "", valid_freq: int = 1):
        """
        对模型进行训练，验证数据集优先级为：预设验证文本>训练划分文本>无验证
        :param checkpoint: 模型的检查点
        :param data_fn: 数据文本路径
        :param buffer_size: Dataset加载缓存大小
        :param batch_size: Dataset加载批大小
        :param max_train_data_size: 最大训练数据量
        :param epochs: 执行训练轮数
        :param checkpoint_save_freq: 检查点保存频率
        :param checkpoint_save_size: 检查点最大保存数
        :param save_dir: 历史指标显示图片保存位置
        :param max_valid_data_size: 最大验证数据量
        :param valid_data_split: 用于从训练数据中划分验证数据，默认0.1
        :param valid_data_fn: 验证数据文本路径
        :param valid_freq: 验证频率
        :return: 各训练指标
        """
        print('训练开始，正在准备数据中')
        train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch, checkpoint_prefix = \
            data_utils.load_data(dict_fn=self.dict_fn, data_fn=data_fn, buffer_size=buffer_size,
                                 batch_size=batch_size, checkpoint_dir=self.checkpoint_dir,
                                 max_length=self.max_length, valid_data_split=valid_data_split,
                                 valid_data_fn=valid_data_fn, max_train_data_size=max_train_data_size,
                                 max_valid_data_size=max_valid_data_size)

        checkpoint_queue = deque(maxlen=checkpoint_save_size + 1)  # 用于保存该次训练产生的检查点名
        history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            self._init_loss_accuracy()

            step_loss = 0
            step_accuracy = 0
            batch_sum = 0
            sample_sum = 0

            for (batch, (inp, tar, weight)) in enumerate(train_dataset.take(steps_per_epoch)):
                step_loss, step_accuracy = self._train_step(inp, tar, weight)
                batch_sum = batch_sum + len(inp)
                sample_sum = steps_per_epoch * len(inp)
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)

            step_time = (time.time() - start_time)
            history['accuracy'].append(step_accuracy.numpy())
            history['loss'].append(step_loss.numpy())

            sys.stdout.write(' - {:.4f}s/step - train_loss: {:.4f} - train_accuracy: {:.4f}\n'
                             .format(step_time, step_loss, step_accuracy))
            sys.stdout.flush()

            if (epoch + 1) % checkpoint_save_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                checkpoint_queue.append(tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir))
                if len(checkpoint_queue) == checkpoint_save_size + 1:
                    checkpoint_name = checkpoint_queue[0]
                    os.remove(checkpoint_name + '.index')
                    os.remove(checkpoint_name + '.data-00000-of-00001')

            if valid_dataset is not None and (epoch + 1) % valid_freq == 0:
                valid_loss, valid_accuracy = self._valid_step(valid_dataset=valid_dataset,
                                                              steps_per_epoch=valid_steps_per_epoch)
                history['val_accuracy'].append(valid_accuracy.numpy())
                history['val_loss'].append(valid_loss.numpy())

        self._show_history(history=history, save_dir=save_dir, valid_freq=valid_freq)
        print('训练结束')
        return history

    def _show_history(self, history, save_dir, valid_freq):
        """
        用于显示历史指标趋势以及保存历史指标图表图
        :param history: 历史指标
        :param save_dir: 历史指标显示图片保存位置
        :param valid_freq: 验证频率
        :return: 无返回值
        """
        train_x_axis = [i + 1 for i in range(len(history['loss']))]
        valid_x_axis = [(i + 1) * valid_freq for i in range(len(history['val_loss']))]

        figure, axis = plt.subplots(1, 1)
        tick_spacing = 1
        if len(history['loss']) > 20:
            tick_spacing = len(history['loss']) // 20
        plt.plot(train_x_axis, history['loss'], label='loss', marker='.')
        plt.plot(train_x_axis, history['accuracy'], label='accuracy', marker='.')
        plt.plot(valid_x_axis, history['val_loss'], label='val_loss', marker='.', linestyle='--')
        plt.plot(valid_x_axis, history['val_accuracy'], label='val_accuracy', marker='.', linestyle='--')
        plt.xticks(valid_x_axis)
        plt.xlabel('epoch')
        plt.legend()

        axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        save_path = save_dir + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    def _valid_step(self, valid_dataset, steps_per_epoch):
        """
        对模型进行训练，验证数据集优先级为：预设验证文本>训练划分文本>无验证
        :param valid_dataset: 验证Dataset
        :param steps_per_epoch: 验证数据总共的步数
        :return: 验证的损失和精度
        """
        print("验证轮次")
        start_time = time.time()
        self._init_loss_accuracy()
        step_loss = 0
        step_accuracy = 0
        batch_sum = 0
        sample_sum = 0

        for (batch, (inp, tar)) in enumerate(valid_dataset.take(steps_per_epoch)):
            step_loss, step_accuracy = self._train_step(inp, tar)
            batch_sum = batch_sum + len(inp)
            sample_sum = steps_per_epoch * len(inp)
            print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                  flush=True)

        step_time = (time.time() - start_time)
        sys.stdout.write(' - {:.4f}s/step - valid_loss: {:.4f} - valid_accuracy: {:.4f}\n'
                         .format(step_time, step_loss, step_accuracy))
        sys.stdout.flush()

        return step_loss, step_accuracy

    def respond(self, req: str):
        """
        对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        :param req: 输入的语句
        :return: 系统回复字符串
        """
        # 对req进行初步处理
        tokenizer = load_tokenizer(self.dict_fn)
        inputs, dec_input = data_utils.preprocess_request(sentence=req, tokenizer=tokenizer,
                                                          max_length=self.max_length, start_sign=self.start_sign)
        self.beam_search_container.reset(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.get_search_inputs()

        for t in range(self.max_length):
            predictions = self._create_predictions(inputs, dec_input, t)
            self.beam_search_container.expand(predictions=predictions, end_sign=tokenizer.word_index.get(self.end_sign))
            # 注意了，如果BeamSearch容器里的beam_size为0了，说明已经找到了相应数量的结果，直接跳出循环
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.get_search_inputs()

        beam_search_result = self.beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = tokenizer.sequences_to_texts(temp)
            text[0] = text[0].replace(self.start_sign, '').replace(self.end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
