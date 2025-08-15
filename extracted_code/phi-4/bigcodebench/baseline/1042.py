import smtplib
from email.message import EmailMessage
import getpass

SERVER_ADDRESS = "localhost"
SERVER_PORT = 25
BUFFER_SIZE = 1024
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def task_func(client_socket):
    sender_email = input("Enter your email: ")
    recipient_email = input("Enter recipient's email: ")
    email_password = getpass.getpass("Enter your email password: ")

    message = client_socket.recv(BUFFER_SIZE).decode('utf-8')

    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = "Message from Client"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender_email, email_password)
        server.send_message(msg)