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

def task_func():
    class RequestHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_type, _ = cgi.parse_header(self.headers.get('Content-Type', ''))
            if content_type!= 'application/json':
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'error','message': 'Content-Type header is not application/json'})
                self.wfile.write(response.encode())
                return

            try:
                length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(length)
                data = json.loads(post_data)
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'error','message': 'Invalid JSON'})
                self.wfile.write(response.encode())
                return

            if 'data' not in data:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'error','message': 'No data key in request'})
                self.wfile.write(response.encode())
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps(SUCCESS_RESPONSE)
            self.wfile.write(response.encode())

    return RequestHandler