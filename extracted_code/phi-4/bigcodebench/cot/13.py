import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            filenames = ftp.nlst()
    except ftplib.all_errors as e:
        if isinstance(e, ftplib.error_perm) and str(e).startswith('550'):
            raise Exception(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")
        elif isinstance(e, ftplib.error_perm) and str(e).startswith('530'):
            raise Exception(f"Failed to log into FTP server {ftp_server} with user {ftp_user}: {str(e)}")
        else:
            raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")

    for filename in filenames:
        local_filename = os.path.join(os.getcwd(), filename)
        try:
            subprocess.run(
                ['wget', '-q', '-O', local_filename, f"ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename}"],
                check=True
            )
        except subprocess.CalledProcessError:
            pass

    return filenames