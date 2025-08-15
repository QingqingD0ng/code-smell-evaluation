import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    filenames = []

    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(user=ftp_user, passwd=ftp_password)
            ftp.cwd(ftp_dir)
            filenames = ftp.nlst()

    except ftplib.all_errors as e:
        if isinstance(e, ftplib.error_perm):
            if str(e).startswith('530'):
                raise Exception(f"Failed to log into FTP server {ftp_server} with user {ftp_user}: {str(e)}")
            elif str(e).startswith('550'):
                raise Exception(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")
        else:
            raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")

    local_dir = os.getcwd()
    for filename in filenames:
        wget_command = [
            'wget', '-q', '-r', '-N', '-P', local_dir,
            f'ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename}'
        ]
        subprocess.run(wget_command, check=True)

    return filenames