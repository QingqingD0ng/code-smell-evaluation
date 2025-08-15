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

class DataHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            if 'data' not in data:
                response = ERROR_RESPONSE.copy()
                response['message'] = 'No data key in request'
            else:
                response = SUCCESS_RESPONSE.copy()
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except json.JSONDecodeError:
            response = ERROR_RESPONSE.copy()
            response['message'] = 'Invalid JSON'
        
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

def run_server(port=8000):
    server_address = ('', port)
    httpd = http.server.HTTPServer(server_address, DataHandler)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()