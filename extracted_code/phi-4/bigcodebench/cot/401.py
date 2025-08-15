from flask import Flask
import os
from flask_mail import Mail

def task_func(app_name):
    app = Flask(app_name)
    mail = Mail(app)

    mail_configs = {
        'MAIL_SERVER': os.getenv('MAIL_SERVER', 'localhost'),
        'MAIL_PORT': int(os.getenv('MAIL_PORT', 25)),
        'MAIL_USE_TLS': os.getenv('MAIL_USE_TLS', 'False').lower() in ['true', '1', 't'],
        'MAIL_USERNAME': os.getenv('MAIL_USERNAME', None),
        'MAIL_PASSWORD': os.getenv('MAIL_PASSWORD', None)
    }

    app.config.update(mail_configs)

    return mail, mail_configs