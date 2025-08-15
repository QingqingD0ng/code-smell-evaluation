from flask import Flask

from flask_mail import Mail

import os


def task_func(app_name):
    app = Flask(app_name)
    mail_server = os.getenv('MAIL_SERVER', 'localhost')
    mail_port = int(os.getenv('MAIL_PORT', '25'))
    use_tls = bool(os.getenv('MAIL_USE_TLS', 'False').lower() == 'true')
    mail_username = os.getenv('MAIL_USERNAME', None)
    mail_password = os.getenv('MAIL_PASSWORD', None)
    
    mail = Mail(app)
    app.config.update(
        MAIL_SERVER=mail_server,
        MAIL_PORT=mail_port,
        MAIL_USE_TLS=use_tls,
        MAIL_USERNAME=mail_username,
        MAIL_PASSWORD=mail_password
    )
    
    return mail, app.extensions.get('mail').mail


# Example usage:

# mail, configs = task_func("test")
# print(mail.app.name)