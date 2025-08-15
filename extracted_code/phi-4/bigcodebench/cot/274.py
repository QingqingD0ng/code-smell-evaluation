import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

class EmailHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, smtp_server, smtp_port, smtp_username, smtp_password, **kwargs):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            email_data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
            return
        
        required_keys = ['subject','message', 'to']
        if not all(key in email_data for key in required_keys):
            self.send_error(400, 'Missing required fields')
            return
        
        subject = email_data['subject']
        message = email_data['message']
        to_email = email_data['to']

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                msg = MIMEText(message)
                msg['Subject'] = subject
                msg['From'] = self.smtp_username
                msg['To'] = to_email
                server.sendmail(self.smtp_username, to_email, msg.as_string())
        except smtplib.SMTPAuthenticationError:
            self.send_error(535, 'Authentication Failed')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', '0')
        self.end_headers()

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    class CustomHandler(EmailHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, smtp_server=smtp_server, smtp_port=smtp_port,
                             smtp_username=smtp_username, smtp_password=smtp_password, **kwargs)

    return CustomHandler