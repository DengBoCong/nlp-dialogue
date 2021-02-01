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

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
# from dialogue.tensorflow.loader import load_seq2seq
# from dialogue.tensorflow.loader import load_smn
# from dialogue.tensorflow.loader import load_transformer

# transformer = load_transformer(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\transformer.json")
# seq2seq = load_seq2seq(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\seq2seq.json")
# smn = load_smn(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\smn.json")

application = Flask(__name__, static_url_path="/static")


@application.route("/")
def home():
    return render_template("/index.html")


@application.route("/message", methods=['POST'])
def response():
    data = request.get_json(silent=True)
    re = data['name']
    # response = transformer.inference(request=re, beam_size=3)
    res = "哈哈哈"
    return res


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8808)
