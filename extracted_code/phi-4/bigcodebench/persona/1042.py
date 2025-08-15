import smtplib
from email.message import EmailMessage
import getpass
import socket

SERVER_ADDRESS = "localhost"
SERVER_PORT = 25
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def task_func(client_socket):
    try:
        # Receive message from client
        message = client_socket.recv(BUFFER_SIZE).decode('utf-8')
        
        # Get sender and recipient email addresses
        sender_email = input("Enter your email: ")
        recipient_email = input("Enter recipient's email: ")
        
        # Get sender's email password securely
        sender_password = getpass.getpass(prompt="Enter your email password: ")
        
        # Create the email message
        email_message = EmailMessage()
        email_message.set_content(message)
        email_message['Subject'] = "Email from Client"
        email_message['From'] = sender_email
        email_message['To'] = recipient_email
        
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(email_message)
        
        print("Email sent successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()

# Example usage:
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((SERVER_ADDRESS, SERVER_PORT))
# server_socket.listen(5)
# client_socket, addr = server_socket.accept()
# task_func(client_socket)