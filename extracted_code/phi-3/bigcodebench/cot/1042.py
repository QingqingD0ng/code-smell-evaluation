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
    client_socket.settimeout(10)
    try:
        message_data = client_socket.recv(BUFFER_SIZE).decode()
    except socket.timeout:
        print("No message received within the timeout period.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    sender_email = input("Enter sender's email: ")
    recipient_email = input("Enter recipient's email: ")
    password = getpass.getpass("Enter sender's email password: ")

    email_message = EmailMessage()
    email_message.set_content(message_data)
    email_message['Subject'] = "Client's Message"
    email_message['From'] = sender_email
    email_message['To'] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(email_message)
            print("Email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("Authentication failed. Please check the sender's email and password.")
    except smtplib.SMTPException as e:
        print(f"An SMTP error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_ADDRESS, SERVER_PORT))
    server_socket.listen(5)
    print(f"Listening on {SERVER_ADDRESS}:{SERVER_PORT}")