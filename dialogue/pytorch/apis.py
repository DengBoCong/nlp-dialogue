#! -*- coding: utf-8 -*-
""" PyTorch Server Api
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Blueprint
from flask_login import login_required

torch_apis = Blueprint("apis", __name__, url_prefix="/apis/torch")


@torch_apis.route('test', methods=['GET', 'POST'])
@login_required
def torch_test():
    print("torch_test")
