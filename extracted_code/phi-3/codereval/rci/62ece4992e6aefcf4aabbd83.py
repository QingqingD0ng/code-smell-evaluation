import subprocess
from typing import List, Tuple, Optional

class CommandExecutionError(Exception):
    def __init__(self, returncode: int, stdout: Optional[str], stderr: Optional[str]):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        message = f"Command execution failed with return code {returncode}. Error output:\n{stderr}" if stderr else f"Command execution failed with return code {returncode}."
        super().__init__(message)

def run_command(commands: List[str], args: List[List[str]], cwd: Optional[str] = None, verbose: bool = False, hide_stderr: bool = False, env: Optional[dict] = None) -> List[Tuple[Optional[str], int]]:
    results = []

    for command, cmd_args in zip(commands, args):
        if not isinstance(cmd_args, list):
            raise TypeError(f"Invalid command or arguments format: {cmd_args}")

        full_command = [command] + cmd_args
        try:
            result = subprocess.run(
                full_command,
                cwd=cwd,
                stdout=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL,
                stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL,
                env=env,
                text=True,
                check=True
            )

            if verbose:
                print(result.stdout)
            results.append((result.stdout, result.returncode))
        except subprocess.CalledProcessError as e:
            error_message = f"Command '{command}' failed with return code {e.returncode}. Error output:\n{e.stderr}" if e.stderr else f"Command '{command}' failed with return code {e.returncode}."
            raise CommandExecutionError(e.returncode, None, error_message)

    return results