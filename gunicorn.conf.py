#! -*- coding: utf-8 -*-
""" Project Server Entrance
"""
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: Apache-2.0 License

workers = 5
threads = 2
daemon = 'false'
worker_class = 'gevent'  # 采用gevent库，支持异步处理请求，提高吞吐量
bind = '0.0.0.0:8000'
worker_connections = 2000
pidfile = 'var/run/gunicorn.pid'
accesslog = 'logs/run/gunicorn_acess.log'
errorlog = 'logs/run/gunicorn_error.log'
loglevel = 'warning'
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
