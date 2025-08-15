import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

class EmailPostHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        email_data = json.loads(self.rfile.read(content_length).decode())
        
        try:
            if not all(key in email_data for key in ('subject','message', 'to')):
                self.send_error(400, 'Bad Request: Missing required fields')
                return
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                try:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    msg = MIMEText(email_data['message'])
                    msg['Subject'] = email_data['subject']
                    msg['To'] = email_data['to']
                    server.sendmail(smtp_username, email_data['to'], msg.as_string())
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.send_header('Content-length', str(len(self.response_page)))
                    self.end_headers()
                    self.wfile.write(self.response_page.encode())
                except smtplib.SMTPAuthenticationError:
                    self.send_error(535, 'Authentication failed')
        except json.JSONDecodeError:
            self.send_error(400, 'Bad Request: Invalid JSON')

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    return EmailPostHandler