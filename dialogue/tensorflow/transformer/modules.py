import time
import tensorflow as tf
from dialogue.tensorflow.load_dataset import load_data
from dialogue.tensorflow.optimizers import loss_func_mask
from dialogue.tools import ProgressBar
from dialogue.tools import show_history


def train(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.optimizers.Adam, epochs: int,
          train_loss: tf.keras.metrics.Mean, train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
          checkpoint: tf.train.CheckpointManager, train_data_path: str, batch_size: int, buffer_size: int,
          max_length: int, checkpoint_save_freq: int, dict_path: str = "", valid_data_split: float = 0.0,
          valid_data_path: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0,
          history_img_path: str = ""):
    """ 训练模块

    :param encoder: encoder模型
    :param decoder: decoder模型
    :param optimizer: 优化器
    :param epochs: 训练周期
    :param train_loss: 损失计算器
    :param train_accuracy: 精度计算器
    :param checkpoint: 检查点管理器
    :param train_data_path: 文本数据路径
    :param batch_size: Dataset加载批大小
    :param buffer_size: Dataset加载缓存大小
    :param max_length: 最大句子长度
    :param checkpoint_save_freq: 检查点保存频率
    :param dict_path: 字典路径，若使用phoneme则不用传
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :param history_img_path: 历史指标数据图表保存路径
    :return: 返回历史指标数据
    """
    print('训练开始，正在准备数据中')
    train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch = \
        load_data(dict_path=dict_path, train_data_path=train_data_path, buffer_size=buffer_size,
                  batch_size=batch_size, max_length=max_length, valid_data_split=valid_data_split,
                  valid_data_path=valid_data_path, max_train_data_size=max_train_data_size,
                  max_valid_data_size=max_valid_data_size)

    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        progress_bar = ProgressBar(total=steps_per_epoch, num=batch_size)

        for (batch, (inp, target, weight)) in enumerate(train_dataset.take(steps_per_epoch)):
            step_loss, step_accuracy = _train_step(encoder=encoder, decoder=decoder, optimizer=optimizer,
                                                   train_loss=train_loss, train_accuracy=train_accuracy,
                                                   inp=inp, target=target, weight=weight)
            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                         .format(step_loss, step_accuracy))

        progress_bar.done(step_time=time.time() - start_time)

        history['accuracy'].append(step_accuracy.numpy())
        history['loss'].append(step_loss.numpy())

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()

            if valid_steps_per_epoch == 0 or valid_dataset is None:
                print("验证数据量过小，小于batch_size，已跳过验证轮次")
            else:
                valid_loss, valid_accuracy = _valid_step(encoder=encoder, decoder=decoder, train_loss=train_loss,
                                                         train_accuracy=train_accuracy, dataset=valid_dataset,
                                                         steps_per_epoch=valid_steps_per_epoch, batch_size=batch_size)
                history['val_accuracy'].append(valid_accuracy.numpy())
                history['val_loss'].append(valid_loss.numpy())

    show_history(history=history, save_dir=history_img_path, valid_freq=checkpoint_save_freq)
    print('训练结束')
    return history


def _train_step(encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.optimizers.Adam,
                train_loss: tf.keras.metrics.Mean, train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
                inp: tf.Tensor, target: tf.Tensor, weight: tf.Tensor = None):
    """训练步

    :param encoder: encoder模型
    :param decoder: decoder模型
    :param optimizer: 优化器
    :param train_loss: 损失计算器
    :param train_accuracy: 精度计算器
    :param inp: 输入序列
    :param target: 目标序列
    :param weight: 样本权重序列
    :return: 返回训练损失和精度
    """
    target_input = target[:, :-1]
    target_real = target[:, 1:]
    with tf.GradientTape() as tape:
        encoder_outputs, padding_mask = encoder(inputs=inp)
        predictions = decoder(inputs=[target_input, encoder_outputs, padding_mask])
        loss = loss_func_mask(target_real, predictions, weight)
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    train_loss(loss)
    train_accuracy(target_real, predictions)

    return train_loss.result(), train_accuracy.result()


def _valid_step(encoder: tf.keras.Model, decoder: tf.keras.Model, train_loss: tf.keras.metrics.Mean,
                train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy, dataset: tf.data.Dataset,
                steps_per_epoch: int, batch_size: int):
    """ 验证步

    :param encoder: encoder模型
    :param decoder: decoder模型
    :param train_loss: 损失计算器
    :param train_accuracy: 精度计算器
    :param dataset: 验证数据集
    :param steps_per_epoch: 验证训练步
    :param batch_size: Dataset加载批大小
    :return: 返回验证损失和精度
    """
    print("验证轮次")
    start_time = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    progress_bar = ProgressBar(total=steps_per_epoch, num=batch_size)

    for (batch, (inp, target)) in enumerate(dataset.take(steps_per_epoch)):
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        encoder_outputs, padding_mask = encoder(inputs=inp)
        predictions = decoder(inputs=[target_input, encoder_outputs, padding_mask])
        loss = loss_func_mask(target_real, predictions)

        train_loss(loss)
        train_accuracy(target_real, predictions)

        progress_bar(current=batch + 1, metrics="- train_loss: {:.4f} - train_accuracy: {:.4f}"
                     .format(train_loss.result(), train_accuracy.result()))

    progress_bar.done(step_time=time.time() - start_time)

    return train_loss.result(), train_accuracy.result()
