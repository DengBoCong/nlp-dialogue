# Copyright 2021 DengBoCong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""web端服务相关模块
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import tensorflow as tf
from dialogue.tensorflow.transformer.model import encoder
from dialogue.tensorflow.transformer.model import decoder
from dialogue.tensorflow.utils import load_checkpoint
from dialogue.tensorflow.transformer.modules import TransformerModule
from dialogue.tensorflow.utils import load_tokenizer
from dialogue.tensorflow.utils import preprocess_request
from dialogue.tensorflow.beamsearch import BeamSearch


def _inference_one_step(decoder, dec_input: tf.Tensor, enc_output: tf.Tensor, padding_mask: tf.Tensor):
    """ 单个推断步

    :param dec_input: decoder输入
    :param enc_output: encoder输出
    :param padding_mask: encoder的padding mask
    :return: 单个token结果
    """
    predictions = decoder(inputs=[dec_input, enc_output, padding_mask])
    predictions = tf.nn.softmax(predictions, axis=-1)
    predictions = predictions[:, -1:, :]
    predictions = tf.squeeze(predictions, axis=1)

    return predictions


def inference(encoder, decoder, request: str, beam_size: int, dict_path, max_sentence,
              start_sign: str = "<start>", end_sign: str = "<end>") -> str:
    """ 对话推断模块

    :param request: 输入句子
    :param beam_size: beam大小
    :param start_sign: 句子开始标记
    :param end_sign: 句子结束标记
    :return: 返回历史指标数据
    """
    tokenizer = load_tokenizer(dict_path)

    enc_input = preprocess_request(sentence=request, tokenizer=tokenizer,
                                   max_length=max_sentence, start_sign=start_sign, end_sign=end_sign)
    enc_output, padding_mask = encoder(inputs=enc_input)
    dec_input = tf.expand_dims([tokenizer.word_index.get(start_sign)], 0)

    beam_search_container = BeamSearch(beam_size=beam_size, max_length=max_sentence, worst_score=0)
    beam_search_container.reset(enc_output=enc_output, dec_input=dec_input, remain=padding_mask)
    enc_output, dec_input, padding_mask = beam_search_container.get_search_inputs()

    for t in range(max_sentence):
        predictions = _inference_one_step(decoder=decoder, dec_input=dec_input,
                                          enc_output=enc_output, padding_mask=padding_mask)

        beam_search_container.expand(predictions=predictions, end_sign=tokenizer.word_index.get(end_sign))
        # 注意了，如果BeamSearch容器里的beam_size为0了，说明已经找到了相应数量的结果，直接跳出循环
        if beam_search_container.beam_size == 0:
            break
        enc_output, dec_input, padding_mask = beam_search_container.get_search_inputs()

    beam_search_result = beam_search_container.get_result(top_k=3)
    result = ""
    # 从容器中抽取序列，生成最终结果
    for i in range(len(beam_search_result)):
        temp = beam_search_result[i].numpy()
        text = tokenizer.sequences_to_texts(temp)
        text[0] = text[0].replace(start_sign, "").replace(end_sign, "").replace(" ", "")
        result = "<" + text[0] + ">" + result
    return result


encoder = encoder(vocab_size=1500, num_layers=2, units=512, embedding_dim=256, num_heads=8, dropout=0.1)
decoder = decoder(vocab_size=1500, num_layers=2, units=512, embedding_dim=256, num_heads=8, dropout=0.1)

checkpoint_manager = load_checkpoint(
    checkpoint_dir=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\checkpoints\tensorflow\transformer",
    execute_type="chat", encoder=encoder, decoder=decoder, checkpoint_save_size=None
)

application = Flask(__name__, static_url_path="/static")


@application.route("/")
def home():
    return render_template("/index.html")


@application.route("/message", methods=['POST'])
def response():
    data = request.get_json(silent=True)
    re = data['name']
    response = inference(encoder=encoder, decoder=decoder, request=re, beam_size=3,
                         dict_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\data\preprocess\transformer_dict.json",
                         max_sentence=40)
    return response


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8808)
