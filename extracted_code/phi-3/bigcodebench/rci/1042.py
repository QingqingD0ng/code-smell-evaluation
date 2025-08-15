import socket
import smtplib
from email.message import EmailMessage
from getpass import getpass
import re
import logging

logging.basicConfig(level=logging.INFO)

SERVER_ADDRESS = 'localhost'
SERVER_PORT = 25
BUFFER_SIZE = 1024
SMTP_SERVER ='smtp.gmail.com'
SMTP_PORT = 587

def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def prompt_for_credentials():
    sender_email = input("Enter sender's email: ")
    password = getpass("Enter sender's email password: ")
    if not validate_email(sender_email) or not password:
        logging.error("Invalid email or password.")
        return None, None
    return sender_email, password

def send_email(sender_email, recipient_email, password):
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = "Message from Client"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    try:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        logging.info("Message sent successfully.")
    except smtplib.SMTPException as e:
        logging.error(f"Failed to send email: {e}")
    finally:
        server.quit()

def task_func(client_socket):
    try:
        message = client_socket.recv(BUFFER_SIZE).decode()
        sender_email, password = prompt_for_credentials()
        if not sender_email or not password:
            return
        recipient_email = input("Enter recipient's email: ")
        if not validate_email(recipient_email):
            logging.error("Invalid recipient email.")
            return
        send_email(sender_email, recipient_email, password)
    except Exception