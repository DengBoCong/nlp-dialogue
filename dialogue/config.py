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
""" 抽象配置类及个模型默认配置类
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

common = {}

transformer = {}

class Config(object):
    def __str__(self):
        print()

    __repr__ = __str__

    def __getitem__(self, item):
        return a + b
