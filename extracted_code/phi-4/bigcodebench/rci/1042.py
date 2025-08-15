import smtplib
from email.message import EmailMessage
import getpass
import socket
import logging
from email.utils import parseaddr
import re

SERVER_ADDRESS = "localhost"
SERVER_PORT = 25
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def receive_message(client_socket):
    return client_socket.recv(1024).decode('utf-8')

def create_email_message(sender, recipient, message):
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = "Message from Client"
    msg['From'] = sender
    msg['To'] = recipient
    return msg

def send_email(sender, recipient, password, message):
    msg = create_email_message(sender, recipient, message)
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def task_func(client_socket):
    sender_email = input("Enter your email: ")
    recipient_email = input("Enter recipient's email: ")

    if not is_valid_email(sender_email) or not is_valid_email(recipient_email):
        logging.error("Invalid email format.")
        return

    email_password = getpass.getpass("Enter your email password: ")

    try:
        message = receive_message(client_socket)
        send_email(sender_email, recipient_email, email_password, message)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_ADDRESS, SERVER_PORT))
    server_socket.listen(5)
    client_socket, addr = server_socket.accept()
    task_func(client_socket)