import tensorflow as tf


def create_padding_mask(seq: tf.Tensor, d_type: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """
    用于创建输入序列的扩充部分的mask
    :param seq: 输入序列
    :param d_type: 运算精度
    :return: mask
    """
    seq = tf.cast(tf.math.equal(seq, 0), d_type)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(seq: tf.Tensor) -> tf.Tensor:
    """
    用于创建当前点以后位置部分的mask
    :param seq: 输入序列
    :return: mask
    """
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask
