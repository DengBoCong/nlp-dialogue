#! -*- coding: utf-8 -*-
""" Server启动入口
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
from configs import create_app
from configs import db
from flask import g
from flask import render_template
from flask_migrate import Migrate
from flask_script import Manager
from flask_script import Shell
from configs import DIALOGUE_APIS_MODULE

application = create_app(config_name=os.environ.get("ENV") or "default")
module = importlib.import_module(DIALOGUE_APIS_MODULE.get(os.environ.get("DIALOGUE_MODULE") or "tf"))
application.register_blueprint(module.apis)

migrate = Migrate(application, db)
server = Manager(application)

with application.app_context():
    g.contextPath = ""


@application.errorhandler(404)
def route_not_found(e):
    """ Api/路由未找到指引
    """
    return render_template("error/404.html"), 404


@application.teardown_appcontext
def shutdown_session(exception=None):
    """ 当Sever销毁时进行的最后操作
    """
    db.session.remove()
    # TODO: send mail while application shutdown


@server.command
def check():
    """ 启动前检查指令
    """
    if not os.path.exists("logs/run"):
        os.mkdir("logs/run")
    # TODO: check system integrity


def make_shell_context():
    """ 使得可以通过shell指令来操控已运行的Server程序
    """
    return dict(app=application, db=db)


if __name__ == "__main__":
    server.add_command("shell", Shell(make_context=make_shell_context()))
    server.run()
