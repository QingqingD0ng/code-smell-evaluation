import subprocess

def subprocess_run_helper(func, *args, timeout, extra_env=None):
    cmd = [func] + list(args)
    env = dict(os.environ, **(extra_env if extra_env else {}))
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout, check=True, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"Command '{cmd}' timed out after {timeout} seconds.")
        result = subprocess.CompletedProcess(args, returncode=1, stdout=None, stderr=None)
    except subprocess.CalledProcessError as e:
        print(f"Command '{cmd}' returned non-zero exit status {e.returncode}.")
        result = subprocess.CompletedProcess(args, returncode=e.returncode, stdout=e.output, stderr=e.stderr)
    return result