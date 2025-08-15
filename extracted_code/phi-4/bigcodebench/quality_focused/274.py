import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

class EmailHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        for key in ['subject','message', 'to']:
            if key not in data:
                self.send_error(400, f"Missing {key} in JSON")
                return

        try:
            mime_message = MIMEText(data['message'])
            mime_message['Subject'] = data['subject']
            mime_message['From'] = 'noreply@example.com'
            mime_message['To'] = data['to']

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(mime_message)
        except smtplib.SMTPAuthenticationError:
            self.send_error(535, "Authentication Failed")
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Email sent successfully")

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    return lambda: EmailHandler(
        request_handler=EmailHandler,
        server_class=http.server.HTTPServer,
        handler_class=EmailHandler
    )