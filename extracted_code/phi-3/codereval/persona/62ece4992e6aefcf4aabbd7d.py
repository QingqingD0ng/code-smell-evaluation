import subprocess
import sys

def subprocess_run_helper(func, *args, timeout, extra_env=None):
    env = extra_env if extra_env else {}
    env.update(os.environ)  # Preserve the current environment variables
    cmd = [sys.executable, '-m', func.__module__, func.__name__] + list(args)
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout, check=True, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        print(f"Command '{cmd}' timed out after {timeout} seconds", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command '{cmd}' returned non-zero exit status", file=sys.stderr)
        sys.exit(e.returncode)
    return result