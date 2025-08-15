from flask import Flask
from flask_mail import Mail
import os

def task_func(app_name):
    app = Flask(app_name)
    mail = Mail(app)

    mail_server = os.getenv('MAIL_SERVER', 'localhost')
    mail_port = int(os.getenv('MAIL_PORT', 25))
    mail_use_tls = bool(int(os.getenv('MAIL_USE_TLS', 0)))
    mail_username = os.getenv('MAIL_USERNAME', None)
    mail_password = os.getenv('MAIL_PASSWORD', None)

    mail.server = mail_server
    mail.port = mail_port
    mail.use_ssl = mail_use_tls
    mail.username = mail_username
    mail.password = mail_password

    return mail, mail.__getattribute__('config').__dict__