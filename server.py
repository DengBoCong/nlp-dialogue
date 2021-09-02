#! -*- coding: utf-8 -*-
""" Project Server Entrance
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
from constant import DIALOGUE_APIS_MODULE

app = create_app(config_name=os.environ.get("ENV") or "default")

# TODO: register route; import_moudle

migrate = Migrate(app, db)
server = Manager(app)

with app.app_context():
    g.contextPath = ""


@app.errorhandler(404)
def page_not_found(e):
    return render_template("error/404.html"), 404


@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()
    # TODO: send mail while app shutdown


@server.command
def check():
    if not os.path.exists("logs/run"):
        os.mkdir("logs/run")
    # TODO: check system integrity


def make_shell_context():
    return dict(app=app, db=db)


if __name__ == "__main__":
    server.add_command("shell", Shell(make_context=make_shell_context()))
    server.run()
