import http.server
import json
from urllib.parse import parse_qs

SUCCESS_RESPONSE = {'status':'success','message': 'Data received successfully.'}
ERROR_RESPONSE = {'status': 'error','message': 'Invalid data received.'}

class JsonRequestHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = parse_qs(self.rfile.read(content_length).decode('utf-8'))
        content_type = self.headers['Content-Type']
        if 'application/json' not in content_type:
            self.send_error(400, 'Content-Type header is not application/json')
            return
        try:
            data = json.loads(post_data['data'][0])
            if 'data' not in data:
                self.send_error(400, 'No data key in request')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(SUCCESS_RESPONSE).encode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = http.server.HTTPServer(server_address, JsonRequestHandler)
    print('Starting JSON HTTP server on port 8000')
    httpd.serve_forever()