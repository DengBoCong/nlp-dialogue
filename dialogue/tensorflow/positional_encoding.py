import numpy as np
import tensorflow as tf
from typing import Tuple


def _get_angles(pos: tf.Tensor, i: tf.Tensor, d_model: tf.Tensor) -> Tuple:
    """pos/10000^(2i/d_model)

    :param pos: 字符总的数量按顺序递增
    :param i: 词嵌入大小按顺序递增
    :param d_model: 词嵌入大小
    :return: shape=(pos.shape[0], d_model)
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int, d_type: tf.dtypes.DType = tf.float32) -> Tuple:
    """PE(pos,2i) = sin(pos/10000^(2i/d_model)) | PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

    :param position: 字符总数
    :param d_model: 词嵌入大小
    :param d_type: 运算精度
    """
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=d_type)
