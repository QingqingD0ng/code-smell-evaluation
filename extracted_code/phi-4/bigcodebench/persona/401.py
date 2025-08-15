from flask import Flask
from flask_mail import Mail
import os

def task_func(app_name):
    app = Flask(app_name)

    mail = Mail(app)
    
    mail_config = {
        "MAIL_SERVER": os.getenv("MAIL_SERVER", "localhost"),
        "MAIL_PORT": int(os.getenv("MAIL_PORT", 25)),
        "MAIL_USE_TLS": os.getenv("MAIL_USE_TLS", "False").lower() in ("true", "1", "t"),
        "MAIL_USERNAME": os.getenv("MAIL_USERNAME", None),
        "MAIL_PASSWORD": os.getenv("MAIL_PASSWORD", None)
    }
    
    app.config.update(mail_config)
    
    return mail, app.config

# Example usage:
# mail, configs = task_func("test")
# print(mail.app.name)  # Output: test