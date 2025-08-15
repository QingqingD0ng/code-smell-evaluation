import cgi
import http.server
import json
from urllib.parse import parse_qs

SUCCESS_RESPONSE = {
  'status':'success',
  'message': 'Data received successfully.'
}

ERROR_RESPONSE = {
  'status': 'error',
  'message': 'Invalid data received.'
}

class PostHandler(http.server.BaseHTTPRequestHandler):
    
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        
        if content_type!= 'application/json':
            self.send_error(400, 'Content-Type header is not application/json')
            return
        
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            self.send_error(400, 'Invalid JSON')
            return
        
        if 'data' not in data:
            self.send_error(400, 'No data key in request')
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        self.wfile.write(json.dumps(SUCCESS_RESPONSE).encode())
    
    def send_error(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'error','message': message}).encode())

def task_func():
    return PostHandler