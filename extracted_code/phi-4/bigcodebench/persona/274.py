import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json

def task_func(smtp_server, smtp_port, smtp_username, smtp_password):
    class EmailHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                email_data = json.loads(post_data)
                
                subject = email_data.get('subject')
                message = email_data.get('message')
                to = email_data.get('to')
                
                if not all([subject, message, to]):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'Bad Request: Missing subject, message, or to field.')
                    return

                msg = MIMEText(message)
                msg['Subject'] = subject
                msg['From'] = smtp_username
                msg['To'] = to

                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.sendmail(smtp_username, to, msg.as_string())

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Email sent successfully.')
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Bad Request: Invalid JSON.')
            except smtplib.SMTPAuthenticationError:
                self.send_response(535)
                self.end_headers()
                self.wfile.write(b'Authentication failed.')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f'Internal Server Error: {str(e)}'.encode())

    return EmailHandler