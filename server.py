#! -*- coding: utf-8 -*-
""" Server Entrance
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
    """ Api/Route not found
    """
    return render_template("error/404.html"), 404


@application.teardown_appcontext
def shutdown_session(exception=None):
    """ The last operation performed when the Sever shutdown
    """
    db.session.remove()
    # TODO: send mail while application shutdown


@server.command
def check():
    """ Check instructions before starting
    """
    if not os.path.exists("logs/run"):
        os.mkdir("logs/run")
    # TODO: check system integrity


def make_shell_context():
    """ Make it possible to control the running Server program through shell commands
    """
    return dict(app=application, db=db)


if __name__ == "__main__":
    server.add_command("shell", Shell(make_context=make_shell_context()))
    server.run()
