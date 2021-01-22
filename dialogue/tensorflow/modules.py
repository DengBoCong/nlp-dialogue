import tensorflow as tf


class Modules(object):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, optimizer: tf.optimizers.Adam, epochs: int,
                 train_loss: tf.keras.metrics.Mean, train_accuracy: tf.keras.metrics.SparseCategoricalAccuracy,
                 checkpoint: tf.train.CheckpointManager, train_data_path: str, batch_size: int, buffer_size: int,
                 max_length: int, checkpoint_save_freq: int, dict_path: str = "", valid_data_split: float = 0.0,
                 valid_data_path: str = "", max_train_data_size: int = 0, max_valid_data_size: int = 0,
                 history_img_path: str = ""):
        """
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
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.checkpoint = checkpoint
        self.max_length = max_length
