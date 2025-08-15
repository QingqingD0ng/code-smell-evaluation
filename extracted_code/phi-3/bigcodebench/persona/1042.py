import smtplib
from email.message import EmailMessage
import getpass
import socket

SERVER_ADDRESS = 'localhost'
SERVER_PORT = 25
BUFFER_SIZE = 1024
SMTP_SERVER ='smtp.gmail.com'
SMTP_PORT = 587

def task_func(client_socket):
    message = client_socket.recv(BUFFER_SIZE).decode()
    sender_email = input('Enter sender email: ')
    recipient_email = input('Enter recipient email: ')
    password = getpass.getpass('Enter sender email password: ')
    email = EmailMessage()
    email['Subject'] = 'Client Message'
    email['From'] = sender_email
    email['To'] = recipient_email
    email.set_content(message)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(email)