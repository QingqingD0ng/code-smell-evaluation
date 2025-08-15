from flask import Flask
import os
from flask_mail import Mail

DEFAULT_MAIL_SERVER = 'localhost'
DEFAULT_MAIL_PORT = 25
DEFAULT_MAIL_USE_TLS = False

def task_func(app_name):
    app = Flask(app_name)

    mail_server = os.getenv('MAIL_SERVER', DEFAULT_MAIL_SERVER)
    mail_port_str = os.getenv('MAIL_PORT', DEFAULT_MAIL_PORT)
    mail_use_tls = os.getenv('MAIL_USE_TLS', str(DEFAULT_MAIL_USE_TLS)).lower() in ('true', '1', 't')
    mail_username = os.getenv('MAIL_USERNAME')
    mail_password = os.getenv('MAIL_PASSWORD')

    try:
        mail_port = int(mail_port_str)
    except ValueError:
        mail_port = DEFAULT_MAIL_PORT

    mail_config = {
        'MAIL_SERVER': mail_server,
        'MAIL_PORT': mail_port,
        'MAIL_USE_TLS': mail_use_tls,
        'MAIL_USERNAME': mail_username,
        'MAIL_PASSWORD': mail_password,
    }

    app.config.update(mail_config)

    mail = Mail(app)

    return mail, mail_config