import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    file_list = []
    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            filenames = ftp.nlst()
            for filename in filenames:
                with open(filename, 'wb') as local_file:
                    ftp.retrbinary('RETR' + filename, local_file.write)
                file_list.append(filename)
    except ftplib.all_errors as e:
        if isinstance(e, ftplib.error_perm):
            raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")
        elif isinstance(e, ftplib.error_perm):
            raise Exception(f"Failed to log into FTP server {ftp_server} with user {ftp_user}: {str(e)}")
        elif isinstance(e, ftplib.error_temp):
            raise Exception(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")
    return file_list