#! -*- coding: utf-8 -*-
""" Project Server Config
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid
from datetime import timedelta
from flask import Flask
from flask_caching import Cache
from flask_login import LoginManager
from flask_mail import Mail
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

mail = Mail()
db = SQLAlchemy()
socket_io = SocketIO()
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "views.login"
login_manager.login_message = "Token is invalid, please regain permissions"

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
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
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "mysql://root:Andie130857@localhost:3306/verb?charset=utf8&autocommit=true"


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'mysql://root:Andie130857@localhost:3306/verb?charset=utf8&autocommit=true'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def create_app(config_name):
    app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
    app.config.from_object(config[config_name])
    config[config_name].init_app(app=app)
    app.secret_key = uuid.uuid1().__str__()
    app.jinja_env.variable_start_string = "[["
    app.jinja_env.variable_end_string = "]]"
    cache = Cache(config={"CACHE_TYPE": "simple"})

    db.init_app(app=app)
    mail.init_app(app=app)
    cache.init_app(app=app)
    login_manager.init_app(app=app)
    socket_io.init_app(app=app)

    from app.view import views
    from dialogue.pytorch.apis import apis
    app.register_blueprint(views)
    app.register_blueprint(apis)

    return app
