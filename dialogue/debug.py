import re, collections
import tensorflow as tf


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=r'D:\DengBoCong\Project\hlp\hlp\chat\data\save_model')
    tflite_model = converter.convert()

    with open('D:/DengBoCong/Project/hlp/hlp/chat/data/save_model/model.tflite') as file:
        file.write(tflite_model)
