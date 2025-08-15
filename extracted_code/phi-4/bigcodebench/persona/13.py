import subprocess
import ftplib
import os

def task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    try:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(ftp_user, ftp_password)
    except Exception as e:
        raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")

    try:
        ftp.cwd(ftp_dir)
    except Exception as e:
        raise Exception(f"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}")

    filenames = []
    ftp.retrlines('LIST', filenames.append)

    local_dir = os.path.join(os.getcwd(), ftp_dir.strip('/'))
    os.makedirs(local_dir, exist_ok=True)

    for filename in filenames:
        if filename.endswith('->') or filename.startswith('total '):
            continue
        _, _, name = filename.rpartition(' ')
        if name:
            wget_command = f"wget -q --no-check-certificate --user={ftp_user} --password={ftp_password} ftp://{ftp_server}{ftp_dir}/{name} -P {local_dir}"
            try:
                subprocess.run(wget_command, shell=True, check=True)
            except subprocess.CalledProcessError:
                pass

    ftp.quit()

    return [name for _, _, name in (f.rpartition(' ') for f in filenames) if name]