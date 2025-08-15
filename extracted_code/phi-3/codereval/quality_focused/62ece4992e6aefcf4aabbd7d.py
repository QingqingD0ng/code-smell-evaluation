import subprocess
from typing import Any, Dict, Tuple

def subprocess_run_helper(func: str, *args: Any, timeout: int, extra_env: Dict[str, str] = None) -> Tuple[subprocess.CompletedProcess, Dict[str, Any]]:
    # Build the command to run the function
    command = [func] + list(args)
    
    # Merge the provided environment variables with the system's environment
    env = dict(os.environ, **(extra_env or {}))
    
    # Run the subprocess with the specified function and arguments
    try:
        result = subprocess.run(command, env=env, timeout=timeout, capture_output=True, text=True)
        return result, env
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Command '{' '.join(command)}' timed out after {timeout} seconds") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred while running command '{' '.join(command)}': {e}") from e

# Example usage:
# result, env = subprocess_run_helper('python', '-c', 'print("Hello, World!")', timeout=5, extra_env={'PYTHONPATH': '/path/to/my/modules'})
# print(result.stdout)