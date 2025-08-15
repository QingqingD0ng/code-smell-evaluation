import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

class EmailPOSTHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        email_data = cgi.parse_qs(self.rfile.read(content_length).decode('utf-8'))
        
        try:
            email_json = json.loads(email_data.get('email', '{}'))
            subject = email_json['subject']
            message = email_json['message']
            to = email_json['to']
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode('utf-8'))
            return
        
        if not all(key in email_json for key in ['subject','message', 'to']):
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Missing keys'}).encode('utf-8'))
            return
        
        smtp_server = smtp_port = smtp_username = smtp_password = ''
        handler = task_func(smtp_server, smtp_port, smtp_username, smtp_password)
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                msg = MIMEText(message)
                msg['Subject'] = subject
                msg['To'] = to
                server.send_message(msg)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self