import subprocess
from typing import Sequence

MAX_COMMAND_LENGTH = 1024  # Common default max command length

def xargs(
    cmd: tuple[str, str],
    varargs: Sequence[str],
    color: bool = False,
    target_concurrency: int = 1,
    _max_length: int = MAX_COMMAND_LENGTH,
    **kwargs: dict
) -> None:
    if not varargs:
        raise ValueError("varargs must be a non-empty sequence of strings.")

    if target_concurrency < 1:
        raise ValueError("target_concurrency must be at least 1.")

    if _max_length < 1:
        raise ValueError("_max_length must be at least 1.")

    if color and not kwargs.get('stdout', False):
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.STDOUT

    def run_command(chunk: Sequence[str]) -> subprocess.Popen:
        current_cmd = (cmd[0], cmd[1].format(*chunk))
        return subprocess.Popen(
            current_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
        )

    for i in range(0, len(varargs), _max_length):
        chunk = varargs[i:i + _max_length]
        process = run_command(chunk)
        stdout, stderr = process.communicate()
        if process.returncode!= 0:
            raise subprocess.CalledProcessError(process.returncode, current_cmd, output=stdout)

# Example usage:
# xargs((['ls', '-l'], '{files}'), ['file1.txt', 'file2.txt', 'file3.txt'])