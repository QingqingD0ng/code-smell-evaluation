import subprocess
import psutil
import time

def task_func(process_name: str) -> str:
    try:
        p = psutil.Popen(["cmd", "/c", process_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait(timeout=10)
        if p.returncode!= 0:
            time.sleep(5)
            subprocess.run(["cmd", "/c", process_name], check=True)
            return f"Process found. Restarting {process_name}."
        else:
            return f"Process found. No need to restart."
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        subprocess.run(["cmd", "/c", process_name], check=True)
        return f"Process not found. Starting {process_name}."
    except subprocess.CalledProcessError:
        return f"Process not found. Starting {process_name}."
    except Exception as e:
        return str(e)