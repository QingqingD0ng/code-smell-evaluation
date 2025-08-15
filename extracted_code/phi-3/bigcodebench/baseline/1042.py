import socket
import smtplib
from email.message import EmailMessage
from getpass import getpass

SERVER_ADDRESS = "localhost"
SERVER_PORT = 25
BUFFER_SIZE = 1024
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def task_func(client_socket):
    try:
        message = client_socket.recv(BUFFER_SIZE).decode()
        sender_email = input("Enter sender's email: ")
        recipient_email = input("Enter recipient's email: ")
        password = getpass("Enter sender's email password: ")

        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = "Message from Client"
        msg['From'] = sender_email
        msg['To'] = recipient_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Message sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()