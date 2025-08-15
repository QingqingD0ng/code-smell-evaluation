from flask import Flask
from flask_mail import Mail
import os

def initialize_mail_for_app(app_name):
    app = Flask(app_name)
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'localhost')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 25))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False').lower() in ('true', '1', 't')
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', None)
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', None)

    mail = Mail(app)
    return mail, app.extensions['mail']

# Example usage:
# mail, configs = initialize_mail_for_app("my_flask_app")
# print(mail.app.name)