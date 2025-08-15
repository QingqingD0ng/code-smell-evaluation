import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    files_downloaded = []
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(ftp_user, ftp_password)
            ftp.cwd(ftp_dir)
            print(f"Changed to directory {ftp_dir} on server {ftp_server}")
            for file in ftp.nlst():
                subprocess.run(['wget', '-qO', '-', ftp_server + ftp_dir + '/' + file], stdout=subprocess.PIPE, text=True)
                files_downloaded.append(file)
        return files_downloaded
    except ftplib.all_errors as e:
        raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")
    except ftplib.error_perm as e:
        raise Exception(f"Failed to log into FTP server {ftp_server} with user {ftp_user}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")