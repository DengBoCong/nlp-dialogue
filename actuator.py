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
""" 总执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from argparse import ArgumentParser

def preprocess():
    pass

def train():
    pass

def valid():
    pass

def run():
    pass


def main() -> None:
    parser = ArgumentParser(description="总执行器", usage="第一个参数必须为 --pipeline PIPELINE")
    parser.add_argument("--pipeline", default="chain", type=str, required=True, help="执行模式，preprocess/train/valid/run")

    options = parser.parse_args(sys.argv[1:5])
    if options.version not in ["tf", "torch"] or options.model not in ["transformer", "seq2seq", "smn"]:
        print("actuator.py: error: VERSION: [tf/torch] MODEL: [transformer/seq2seq/smn]")
        exit(0)
    models[options.version][options.model]()


if __name__ == "__main__":
    main()
