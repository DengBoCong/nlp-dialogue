#! -*- coding: utf-8 -*-
""" UserProfile
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from configs import db
from datetime import datetime
from flask_login import UserMixin


class UserProfile(db.Model, UserMixin):
    """ 用户画像
    """
    __tablename__ = "DIALOGUE_USERPROFILE"

    ID = db.Column(db.String(50), primary_key=True, nullable=False, comment="ID")
    CREATE_DATETIME = db.Column(db.DateTime, default=datetime.now(), nullable=False, comment="创建时间")
    LAST_DATETIME = db.Column(db.DateTime, index=True, nullable=False, comment="最后登录时间")
    UPDATE_DATETIME = db.Column(db.DateTime, nullable=False, comment="更新时间")
    EMAIL = db.Column(db.String(60), index=True, nullable=False, default="", unique=True, comment="邮箱账号")
    NAME = db.Column(db.String(50), nullable=False, default="", comment="名称")
    AVATAR_URL = db.Column(db.String(255), nullable=False, default="", comment="图片地址")
    SEX = db.Column(db.String(1), nullable=False, default="0", comment="性别")
    AGE = db.Column(db.Integer, nullable=False, default=0, comment="年龄")
    Contact = db.Column(db.String(30), nullable=False, default="", comment="联系方式")

    def __repr__(self):
        return '<UserProfile name: %s>\n' % self.NAME

    def to_json(self):
        """ UserProfile字符串格式化
        """
        return {
            'ID': self.ID,
            'CREATE_DATETIME': self.CREATE_DATETIME.strftime('%Y-%m-%d %H:%M:%S'),
            'LAST_DATETIME': self.LAST_DATETIME.strftime('%Y-%m-%d %H:%M:%S'),
            'UPDATE_DATETIME': self.UPDATE_DATETIME.strftime('%Y-%m-%d %H:%M:%S'),
            'EMAIL': self.EMAIL,
            'NAME': self.NAME,
            'AVATAR_URL': self.AVATAR_URL,
            'SEX': self.SEX,
            'AGE': self.AGE,
            'Contact': self.Contact,
        }
