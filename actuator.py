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
"""总执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from argparse import ArgumentParser
from dialogue.tensorflow.seq2seq.actuator import tf_seq2seq
from dialogue.tensorflow.smn.actuator import tf_smn
from dialogue.tensorflow.transformer.actuator import tf_transformer


def main() -> None:
    parser = ArgumentParser(description="总执行器", usage="前两个参数必须为 --version VERSION --model MODEL")
    parser.add_argument("--version", default="tf", type=str, required=True, help="执行版本")
    parser.add_argument("--model", default="transformer", type=str, required=True, help="执行模型")

    models = {
        "tf": {
            "transformer": lambda: tf_transformer(),
            "seq2seq": lambda: tf_seq2seq(),
            "smn": lambda: tf_smn(),
        },
        "torch": {
            "transformer": None
        }
    }

    options = parser.parse_args(sys.argv[1:5])
    try:
        models[options.version][options.model]()
    except KeyError:
        print("actuator.py: error: VERSION: [tf/torch] MODEL: [transformer/seq2seq/smn]")


if __name__ == '__main__':
    main()
