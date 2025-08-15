import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    try:
        with ftplib.FTP(ftp_server, ftp_user, ftp_password) as ftp:
            ftp.cwd(ftp_dir)
            filenames = ftp.nlst()
    except ftplib.all_errors as e:
        raise Exception(f"FTP error: {str(e)}")

    downloaded_files = []

    for filename in filenames:
        local_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(local_path):
            try:
                command = ['wget', '-q', '-P', os.getcwd(), f'ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename}']
                subprocess.run(command, check=True)
                downloaded_files.append(filename)
            except subprocess.CalledProcessError:
                print(f"Failed to download {filename}")

    return downloaded_files