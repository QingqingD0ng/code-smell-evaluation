import cgi
import http.server
import json

class JSONRequestHandler(http.server.BaseHTTPRequestHandler):
    SUCCESS_RESPONSE = {
       'status':'success',
       'message': 'Data received successfully.'
    }

    ERROR_RESPONSE = {
       'status': 'error',
       'message': 'Invalid data received.'
    }

    def do_POST(self):
        content_type = self.headers.get('Content-Type')

        if content_type!= 'application/json':
            self.send_error(400, 'Content-Type header is not application/json')
            return
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length)
            data = json.loads(request_body)

            if 'data' not in data:
                self.send_error(400, 'No data key in request')
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps(self.SUCCESS_RESPONSE).encode('utf-8')
            self.wfile.write(response)
        
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
        except ValueError:
            self.send_error(400, 'Invalid Content-Length header')

def task_func():
    return JSONRequestHandler