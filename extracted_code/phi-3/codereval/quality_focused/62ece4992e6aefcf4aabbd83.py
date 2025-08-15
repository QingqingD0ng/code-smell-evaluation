import subprocess
from typing import List, Optional, Dict

def run_command(commands: List[str], args: List[str], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    command = commands[-1]  # Assuming the last command in the list is the one to run
    full_command = [command] + args
    return subprocess.run(full_command, cwd=cwd, stdout=subprocess.PIPE if verbose else None, stderr=subprocess.DEVNULL if hide_stderr else None, env=env)