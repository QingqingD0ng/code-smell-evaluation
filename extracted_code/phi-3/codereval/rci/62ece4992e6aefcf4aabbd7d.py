import os
import subprocess

def subprocess_run_helper(func, *args, timeout, extra_env=None):
    cmd = [func] + list(args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout, check=True, capture_output=True, text=True)
        return result
    except subprocess.TimeoutExpired:
        print(f"Command '{cmd}' timed out after {timeout} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"Command '{cmd}' returned non-zero exit status {e.returncode}.")
    except Exception as e:
        print(f"An error occurred while running command '{cmd}': {str(e)}")

    return subprocess.CompletedProcess(cmd, e.returncode, e.stdout, e.stderr)