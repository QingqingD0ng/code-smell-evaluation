import subprocess
import os
import sys
import time
import logging
from typing import Optional, List

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def task_func(script_path: str, wait: bool = True, *args: str) -> Optional[int]:
    if not os.path.isfile(script_path):
        logging.error(f"The script '{script_path}' does not exist.")
        raise ValueError(f"The script '{script_path}' does not exist.")

    command = [sys.executable, script_path] + list(args)
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
        if wait:
            return result.returncode
        else:
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Script '{script_path}' failed with return code {e.returncode}.\nOutput: {e.output}")
        raise subprocess.CalledProcessError(e.returncode, e.cmd) from e
    except subprocess.TimeoutExpired:
        logging.error(f"Script '{script_path}' timed out after {e.timeout} seconds.")
        raise subprocess.TimeoutExpired("Script execution timed out.") from e