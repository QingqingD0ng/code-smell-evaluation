import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            filenames = ftp.nlst()
    except Exception as e:
        raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")

    for filename in filenames:
        try:
            command = f'wget -q -P {os.getcwd()} ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename}'
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError:
            continue

    return filenames