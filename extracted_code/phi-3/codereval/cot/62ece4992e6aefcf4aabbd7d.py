import subprocess
from typing import Any, Callable, Dict, Optional

def subprocess_run_helper(func: Callable[..., Any], *args: Any, timeout: Optional[int] = None, extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    command = [func.__module__] + list(args)
    env = extra_env.copy() if extra_env else None
    try:
        result = subprocess.run(command, env=env, timeout=timeout, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Function {func.__name__} failed with return code {e.returncode}") from e
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds") from e
    return result