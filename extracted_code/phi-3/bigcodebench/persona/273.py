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

class PostRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = cgi.parse_header(self.headers.get('Content-Type'))[1]
        body = self.rfile.read(content_length).decode('utf-8')
        
        if post_data!= 'application/json':
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Content-Type header is not application/json')
            return
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid JSON')
            return
        
        if 'data' not in data:
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'No data key in request')
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(SUCCESS_RESPONSE).encode('utf-8'))