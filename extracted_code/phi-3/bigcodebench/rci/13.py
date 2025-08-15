import os
import urllib.request
import ftplib
import getpass

def task_func(ftp_server='ftp.dlptest.com', ftp_dir='/ftp/test', download_dir='downloads', ftp_user=None, ftp_password=None):
    filenames = []

    # Prompt for FTP credentials if not provided
    if not ftp_user or not ftp_password:
        ftp_user = input("Enter FTP username: ")
        ftp_password = getpass.getpass("Enter FTP password: ")

    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            for item in ftp.nlst():
                file_path = os.path.join(download_dir, item)
                if os.path.isfile(file_path):
                    with urllib.request.urlopen(ftp.nlist(item)) as source, open(file_path, 'wb') as dest:
                        dest.write(source.read())
                        filenames.append(item)
                else:
                    print(f"Item '{item}' is not a file.")
    except ftplib.all_errors as e:
        print(f"Failed to connect to FTP server {ftp_server}: {str(e)}")
    except Exception as e:
        print(f"Failed to download items: {str(e)}")

    return filenames

# Example usage
task_func()