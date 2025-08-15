import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    class EmailHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = cgi.parse_qs(self.rfile.read(content_length).decode())

            try:
                email_data = json.loads(post_data.get('email', [{}])[0])
                subject = email_data.get('subject')
                message = email_data.get('message')
                recipient = email_data.get('to')

                if not subject or not message or not recipient:
                    self.send_response(400)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"Missing'subject','message', or 'to' keys in email data")
                    return

                msg = MIMEText(message)
                msg['Subject'] = subject
                msg['From'] = smtp_username
                msg['To'] = recipient

                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.send_message(msg)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', '0')
                self.end_headers()

            except json.JSONDecodeError:
                logging.error("Invalid JSON in email data")
                self.send_response(400)
                self.send_