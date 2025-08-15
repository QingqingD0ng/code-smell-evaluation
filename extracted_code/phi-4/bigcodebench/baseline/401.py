from flask import Flask
import os
from flask_mail import Mail

def task_func(app_name):
    app = Flask(app_name)

    mail_config = {
        'MAIL_SERVER': os.getenv('MAIL_SERVER', 'localhost'),
        'MAIL_PORT': int(os.getenv('MAIL_PORT', 25)),
        'MAIL_USE_TLS': os.getenv('MAIL_USE_TLS', 'False') == 'True',
        'MAIL_USERNAME': os.getenv('MAIL_USERNAME'),
        'MAIL_PASSWORD': os.getenv('MAIL_PASSWORD'),
    }

    app.config.update(mail_config)

    mail = Mail(app)

    return mail, mail_config