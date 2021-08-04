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

pipelines = {
    # "preprocess": pa
}

def preprocess():
    pass


def train():
    pass


def valid():
    pass


def run():
    pass


def main() -> None:
    parser = ArgumentParser(description="total actuator", usage="the first parameter must be --pipeline PIPELINE")
    parser.add_argument("--pipeline", default="chain", type=str, required=True,
                        help="execution mode，preprocess/train/valid/run")

    options = parser.parse_args().__dict__

    if not options.get("pipeline"):
        raise AttributeError("actuator.py: error: PIPELINE: [preprocess/train/valid/run]")




if __name__ == "__main__":
    main()
