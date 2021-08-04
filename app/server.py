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

import json
import pysolr
from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS
from dialogue.tensorflow.loader import load_seq2seq
from dialogue.tensorflow.loader import load_smn
from dialogue.tensorflow.loader import load_transformer
from dialogue.tensorflow.modules import Modules

transformer = load_transformer(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\transformer.json")
seq2seq = load_seq2seq(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\seq2seq.json")
smn = load_smn(config_path=r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\smn.json")

application = Flask(__name__, static_url_path="/static")
CORS(application, supports_credentials=True)
application.jinja_env.variable_start_string = "[["
application.jinja_env.variable_end_string = "]]"
history = []  # 用于存放历史对话
solr = pysolr.Solr(url="http://49.235.33.100:8983/solr/smn/", always_commit=True, timeout=10)


def load_running_msg(data: dict, modules: Modules) -> None:
    data["status"] = "error" if modules is None else "processing"


@application.route("/")
def index():
    return render_template("index.html")


@application.route("/transformer", methods=['GET'])
def transformer_page():
    message = {}

    with open(r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\transformer.json",
              "r", encoding="utf-8") as config_file:
        options = json.load(config_file)
    message["options"] = options

    load_running_msg(data=message, modules=transformer)
    message["model"] = "Transformer"

    return message


@application.route("/seq2seq")
def seq2seq_page():
    message = {}

    with open(r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\seq2seq.json",
              "r", encoding="utf-8") as config_file:
        options = json.load(config_file)
    message["options"] = options

    load_running_msg(data=message, modules=seq2seq)
    message["model"] = "Seq2Seq"

    return message


@application.route("/smn")
def smn_page():
    message = {}

    with open(r"D:\DengBoCong\Project\nlp-dialogue\dialogue\config\smn.json",
              "r", encoding="utf-8") as config_file:
        options = json.load(config_file)
    message["options"] = options

    load_running_msg(data=message, modules=smn)
    message["model"] = "SMN"

    return message


@application.route("/transformer/message", methods=['POST'])
def transformer_inference():
    data = request.get_json(silent=True)
    re = data['name']
    response = transformer.inference(request=re, beam_size=3)
    return response


@application.route("/seq2seq/message", methods=['POST'])
def seq2seq__inference():
    data = request.get_json(silent=True)
    re = data['name']
    response = seq2seq.inference(request=re, beam_size=3)
    return response


@application.route("/smn/message", methods=['POST'])
def smn__inference():
    data = request.get_json(silent=True)
    re = data['name']
    history.append(re)
    response = smn.inference(request=history, solr=solr, max_utterance=10)
    history.append(response)
    return response


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8808)
