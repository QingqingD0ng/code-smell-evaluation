import os
import subprocess
import ftplib

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    filenames = []
    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            for item in ftp.nlst():
                if os.path.isfile(item):
                    subprocess.run(['wget', '-P', 'downloads/', item], check=True)
                    filenames.append(item)
    except ftplib.all_errors as e:
        print(f"Failed to connect to FTP server {ftp_server}: {str(e)}")
    except Exception as e:
        print(f"Failed to log into FTP server {ftp_server} with user {ftp_user}: {str(e)}")
    except Exception as e:
        print(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")
    return filenames

task_func()