import smtplib
from email.message import EmailMessage
import getpass


def task_func(client_socket):
    message = client_socket.recv(BUFFER_SIZE).decode()
    
    sender_email = input("Enter sender's email: ")
    recipient_email = input("Enter recipient's email: ")
    password = getpass.getpass("Enter sender's email password: ")
    
    email_message = EmailMessage()
    email_message.set_content(message)
    email_message['Subject'] = "Message from Client"
    email_message['From'] = sender_email
    email_message['To'] = recipient_email
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(email_message)