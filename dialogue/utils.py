import os
import logging


def log_operator(level: str, log_file: str = None,
                 log_format: str = "[%(levelname)s] - [%(asctime)s] - [file: %(filename)s] - "
                                   "[function: %(funcName)s] - [%(message)s]") -> logging.Logger:
    """
    日志操作方法，日志级别有'CRITICAL','FATAL','ERROR','WARN','WARNING','INFO','DEBUG','NOTSET'
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    :param log_file: 日志路径
    :param message: 日志信息
    :param level: 日志级别
    :param log_format: 日志信息格式
    :return: 日志记录器
    """
    if log_file is None:
        log_file = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")] + '\\hlp\\chat\\data\\runtime.log'

    logger = logging.getLogger()
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level=level)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
