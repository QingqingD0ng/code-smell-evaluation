import http.server
import json

SUCCESS_RESPONSE = {
   'status':'success',
   'message': 'Data received successfully.'
}

ERROR_RESPONSES = {
    'unsupported_content_type': {'status': 'error','message': 'Content-Type header is not application/json'},
   'missing_data_key': {'status': 'error','message': 'No data key in request'},
    'invalid_json': {'status': 'error','message': 'Invalid JSON'}
}

HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400

def task_func():
    class JSONPostRequestHandler(http.server.BaseHTTPRequestHandler):
        def send_response_json(self, response, status_code=HTTP_STATUS_OK):
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_POST(self):
            content_type = self.headers.get('Content-Type', '')
            if content_type!= 'application/json':
                self.send_response_json(ERROR_RESPONSES['unsupported_content_type'], HTTP_STATUS_BAD_REQUEST)
                return

            try:
                length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(length)
                data = json.loads(post_data)
            except json.JSONDecodeError:
                self.send_response_json(ERROR_RESPONSES['invalid_json'], HTTP_STATUS_BAD_REQUEST)
                return

            if 'data' not in data:
                self.send_response_json(ERROR_RESPONSES['missing_data_key'], HTTP_STATUS_BAD_REQUEST)
                return

            self.send_response_json(SUCCESS_RESPONSE)

    return JSONPostRequestHandler