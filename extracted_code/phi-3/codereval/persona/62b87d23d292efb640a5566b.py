import subprocess
from typing import List

def run_command(commands: List[str], args: List[str], cwd: str = None, verbose: bool = False, hide_stderr: bool = False, env: dict = None) -> subprocess.CompletedProcess:
    if isinstance(commands, str):
        commands = [commands]
    command_args = commands + args
    if verbose:
        print(f"Running command: {' '.join(command_args)}")
    if cwd:
        return subprocess.run(command_args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL, env=env, check=True)
    else:
        return subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL, env=env, check=True)