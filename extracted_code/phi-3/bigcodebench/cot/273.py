import cgi
import http.server
import json

SUCCESS_RESPONSE = {'status':'success','message': 'Data received successfully.'}
ERROR_RESPONSE = {'status': 'error','message': 'Invalid data received.'}

class DataHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if 'content-type' not in self.headers or self.headers['Content-Type']!= 'application/json':
            self._send_error(400, "Content-Type header is not application/json")
            return

        if 'data' not in post_data:
            self._send_error(400, "No data key in request")
            return

        self._send_response(200, SUCCESS_RESPONSE)

    def _send_response(self, status_code, response_data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json.dumps(response_data).encode('utf-8'))))
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def _send_error(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json.dumps({'status': 'error','message': message}).encode('utf-8'))))
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'error','message': message}).encode('utf-8'))