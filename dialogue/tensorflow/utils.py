import tensorflow as tf


def combine_mask(seq: tf.Tensor, d_type: tf.dtypes.DType = tf.float32):
    """对input中的不能见单位进行mask

    :param seq: 输入序列
    :param d_type: 运算精度
    :return: mask
    """
    look_ahead_mask = _create_look_ahead_mask(seq)
    padding_mask = create_padding_mask(seq, d_type=d_type)
    return tf.maximum(look_ahead_mask, padding_mask)


def create_padding_mask(seq: tf.Tensor, d_type: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """
    用于创建输入序列的扩充部分的mask
    :param seq: 输入序列
    :param d_type: 运算精度
    :return: mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), d_type)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def _create_look_ahead_mask(seq: tf.Tensor) -> tf.Tensor:
    """
    用于创建当前点以后位置部分的mask
    :param seq: 输入序列
    :return: mask
    """
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask


def load_checkpoint(checkpoint_dir: str, execute_type: str, checkpoint_save_size: int, model: tf.keras.Model = None,
                    encoder: tf.keras.Model = None, decoder: tf.keras.Model = None) -> tf.train.CheckpointManager:
    """加载检查点，同时支持Encoder-Decoder结构加载，两种类型的模型二者只能传其一

    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    :param model: 传入的模型
    :param encoder: 传入的Encoder模型
    :param decoder: 传入的Decoder模型
    """
    if model is not None:
        checkpoint = tf.train.Checkpoint(model=model)
    elif encoder is not None and decoder is not None:
        checkpoint = tf.train.CheckpointManager(encoder=encoder, decoder=decoder)
    else:
        print("加载检查点所传入模型有误，请检查后重试！")

    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir,
                                                    max_to_keep=checkpoint_save_size)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    else:
        if execute_type != "train" and execute_type != "pre_treat":
            print("没有检查点，请先执行train模式")
            exit(0)

    return checkpoint_manager
