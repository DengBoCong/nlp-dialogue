#! -*- coding: utf-8 -*-
""" TensorFlow Server Api
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Blueprint

apis = Blueprint("tf_apis", __name__, url_prefix="/apis/tf")


@apis.route('test', methods=['GET', 'POST'])
def test():
    return "tf_apis"

