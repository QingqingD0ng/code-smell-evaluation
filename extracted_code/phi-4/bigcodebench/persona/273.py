import cgi
import http.server
import json

class CustomHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_type, _ = cgi.parse_header(self.headers.get('Content-Type', ''))
        
        if content_type!= 'application/json':
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'error','message': 'Content-Type header is not application/json'}
            self.wfile.write(json.dumps(response).encode())
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'error','message': 'Invalid JSON'}
            self.wfile.write(json.dumps(response).encode())
            return

        if 'data' not in data:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'error','message': 'No data key in request'}
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {'status':'success','message': 'Data received successfully.'}
        self.wfile.write(json.dumps(response).encode())

def task_func():
    return CustomHandler