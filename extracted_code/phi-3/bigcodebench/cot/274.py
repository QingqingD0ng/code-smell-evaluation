import cgi
import http.server
import smtplib
from email.mime.text import MIMEText
import json
import urllib.parse

class EmailPostHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        email_data = json.loads(post_data.decode('utf-8'))
        
        if not all(key in email_data for key in ('subject','message', 'to')):
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Missing required keys in JSON data.')
            return
        
        subject = email_data['subject']
        message = email_data['message']
        to = email_data['to']
        
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = 'your-email@example.com'
            msg['To'] = to
            msg['Content-Type'] = 'text/plain'
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(msg['From'], [msg['To']], msg.as_string())
            server.quit()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status':'success'}).encode('utf-8'))
            
        except smtplib.SMTPAuthenticationError:
            self.send_response(535)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'SMTP Authentication Failed.')
        except json.JSONDecodeError: