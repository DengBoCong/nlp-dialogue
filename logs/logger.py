#! -*- coding: utf-8 -*-
""" log and pipeline data collector
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime


class Collector(object):
    """ used for collect pipeline data, special training logs, inference
        logs, evaluate logs, etc. And provide visual data and log views.
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.collector_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        if os.path.exists(self.collector_dir):
            os.makedirs(self.collector_dir)

    def write_runtime_log(self, file_name: str, line: str):
        """ write runtime log file
        :param file_name: log write location file name
        :param line: log description
        """
        with open(os.path.join(self.collector_dir, "runtime.logs"), 'a', encoding="utf-8") as file:
            file.write("INFO {} {} {}".format(datetime.now(), file_name, line))

    def write_training_log(self, metrics: dict, if_batch_end: bool = False):
        """ write training log file
        """
        pass

    def write_evaluate_log(self, metrics: dict):
        """ write evaluate log file
        """
        pass

        # if os.path.exists(os.path.join(log_dir, ))


# def log_operator(level: str, log_file: str = None,
#                  log_format: str = "[%(levelname)s] - [%(asctime)s] - [file: %(filename)s] - "
#                                    "[function: %(funcName)s] - [%(message)s]") -> logging.Logger:
#     """ 日志操作方法，日志级别有"CRITICAL","FATAL","ERROR","WARN","WARNING","INFO","DEBUG","NOTSET"
#     CRITICAL = 50, FATAL = CRITICAL, ERROR = 40, WARNING = 30, WARN = WARNING, INFO = 20, DEBUG = 10, NOTSET = 0
#
#     :param log_file: 日志路径
#     :param level: 日志级别
#     :param log_format: 日志信息格式
#     :return: 日志记录器
#     """
#     if log_file is None:
#         log_file = os.path.abspath(__file__)[
#                    :os.path.abspath(__file__).rfind("\\dialogue\\")] + "\\dialogue\\data\\preprocess\\runtime.logs"
#
#     logger = logging.getLogger()
#     logger.setLevel(level)
#     file_handler = logging.FileHandler(log_file, encoding="utf-8")
#     file_handler.setLevel(level=level)
#     formatter = logging.Formatter(log_format)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#
#     return logger

if __name__ == "__main__":
    print(os.path.join("D:\\te", "test", "te"))
    print(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
