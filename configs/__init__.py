#! -*- coding: utf-8 -*-
""" 全局配置
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid
from configs.configs import config
from configs.constant import *
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


@login_manager.user_loader
def load_user(user_id):
    """ 会话激活
    """
    return {"ID": "null"}


basedir = os.path.abspath(os.path.dirname(__file__))


def create_app(config_name):
    """ 整合Server app的相关配置
    """
    app = Flask(__name__, template_folder="../app/templates", static_folder="../app/static")
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
