#! -*- coding: utf-8 -*-
""" Session
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from configs import db
from datetime import datetime


class Dialogue(db.Model):
    """ 对话文本
    """
    __tablename__ = "DIALOGUE_DIALOGUE"

    ID = db.Column(db.String(50), primary_key=True, nullable=False, comment="ID")
    CREATE_DATETIME = db.Column(db.DateTime, default=datetime.now(), nullable=False, comment="创建时间")
    EMAIL = db.Column(db.String(60), index=True, nullable=False, default="", unique=True, comment="邮箱账号")
    IDENTITY = db.Column(db.Enum("Agent", "User"), nullable=False, comment="发送者身份")
    UTTERANCE = db.Column(db.String(255), nullable=False, default="", comment="文本内容")

    def __repr__(self):
        return '<Dialogue Email: %s>\n' % self.EMAIL

    def to_json(self):
        """ Dialogue字符串格式化
        """
        return {
            'ID': self.ID,
            'CREATE_DATETIME': self.CREATE_DATETIME.strftime('%Y-%m-%d %H:%M:%S'),
            'EMAIL': self.EMAIL,
            'IDENTITY': self.IDENTITY,
            'UTTERANCE': self.UTTERANCE,
        }
