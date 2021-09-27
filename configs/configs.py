#! -*- coding: utf-8 -*-
""" Server Configuration
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import timedelta
from flask import Flask


class Config:
    """ server configuration
    """
    # session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=3)
    # mail
    MAIL_SERVER = os.environ.get("MAIL_SERVER")
    MAIL_PROT = 25
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD")
    # db
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_POOL_SIZE = 20
    SQLALCHEMY_MAX_OVERFLOW = 10
    SQLALCHEMY_POOL_RECYCLE = 1200

    @classmethod
    def init_app(cls, app: Flask):
        """ common configuration
        """
        app.config["DEBUG"] = cls.DEBUG
        app.config["PERMANENT_SESSION_LIFETIME"] = cls.PERMANENT_SESSION_LIFETIME
        app.config["MAIL_SERVER"] = cls.MAIL_SERVER
        app.config["MAIL_PROT"] = cls.MAIL_PROT
        app.config["MAIL_USE_TLS"] = cls.MAIL_USE_TLS
        app.config["MAIL_USE_SSL"] = cls.MAIL_USE_SSL
        app.config["MAIL_USERNAME"] = cls.MAIL_USERNAME
        app.config["MAIL_PASSWORD"] = cls.MAIL_PASSWORD
        app.config["SQLALCHEMY_DATABASE_URI"] = cls.SQLALCHEMY_DATABASE_URI
        app.config["SQLALCHEMY_COMMIT_ON_TEARDOWN"] = cls.SQLALCHEMY_COMMIT_ON_TEARDOWN
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = cls.SQLALCHEMY_TRACK_MODIFICATIONS
        app.config["SQLALCHEMY_ECHO"] = cls.SQLALCHEMY_ECHO
        app.config["SQLALCHEMY_POOL_SIZE"] = cls.SQLALCHEMY_POOL_SIZE
        app.config["SQLALCHEMY_MAX_OVERFLOW"] = cls.SQLALCHEMY_MAX_OVERFLOW
        app.config["SQLALCHEMY_POOL_RECYCLE"] = cls.SQLALCHEMY_POOL_RECYCLE


class DevelopmentConfig(Config):
    """ development configuration
    """
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "mysql://root:Andie130857@localhost:3306/verb?charset=utf8&autocommit=true"


class ProductionConfig(Config):
    """ production configuration
    """
    SQLALCHEMY_DATABASE_URI = 'mysql://root:Andie130857@localhost:3306/verb?charset=utf8&autocommit=true'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
