import cgi
import http.server
import json

SUCCESS_RESPONSE = {
   'status':'success',
   'message': 'Data received successfully.'
}

ERROR_RESPONSE = {
   'status': 'error',
   'message': 'Invalid data received.'
}

class TaskHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = cgi.parse_header(self.headers['Content-Type'])[1]
        if post_data['type']!= 'application/json':
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Content-Type header is not application/json')
        else:
            try:
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                if 'data' not in data:
                    self.send_response(400)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'No data key in request')
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(SUCCESS_RESPONSE).encode('utf-8'))
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Invalid JSON')