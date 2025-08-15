import smtplib
from email.message import EmailMessage
import getpass
import socket

SERVER_ADDRESS = "localhost"
SERVER_PORT = 25
BUFFER_SIZE = 1024
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def task_func(client_socket):
    data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
    sender_email, recipient_email, password = data.split(',')

    message = EmailMessage()
    message.set_content(data)
    message['Subject'] = 'Received Message'
    message['From'] = sender_email
    message['To'] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(message)
            print("Message sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_ADDRESS, SERVER_PORT))
    server_socket.listen(5)
    client_socket, addr = server_socket.accept()
    task_func(client_socket)