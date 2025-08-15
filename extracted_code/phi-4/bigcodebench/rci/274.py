import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

class EmailHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            email_data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
            return

        subject = email_data.get('subject')
        message = email_data.get('message')
        to = email_data.get('to')

        if not subject or not message or not to:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Missing fields'}).encode())
            return

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = self.server.smtp_username
        msg['To'] = to

        try:
            with smtplib.SMTP(self.server.smtp_server, self.server.smtp_port) as server:
                server.starttls()
                server.login(self.server.smtp_username, self.server.smtp_password)
                server.sendmail(self.server.smtp_username, to, msg.as_string())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'Email sent'}).encode())
        except smtplib.SMTPAuthenticationError:
            self.send_response(535)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Authentication Failed'}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    class CustomEmailHandler(EmailHandler):
        smtp_server = smtp_server
        smtp_port = smtp_port
        smtp_username = smtp_username
        smtp_password = smtp_password

    return CustomEmailHandler