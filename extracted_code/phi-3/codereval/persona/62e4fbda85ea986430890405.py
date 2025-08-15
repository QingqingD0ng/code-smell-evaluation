import subprocess
import os
from typing import Sequence

_platform_max_length = 4096  # Example platform max length, adjust as needed

def _get_platform_max_length() -> int:
    return _platform_max_length

def xargs(cmd: tuple[str,...], varargs: Sequence[str], color: bool = False, target_concurrency: int = 1, **kwargs):
    if not isinstance(cmd, tuple) or not all(isinstance(arg, str) for arg in cmd):
        raise ValueError("cmd must be a tuple of strings")
    if not isinstance(varargs, Sequence) or not all(isinstance(arg, str) for arg in varargs):
        raise ValueError("varargs must be a sequence of strings")
    
    command_str =''.join(cmd)
    varargs_str =''.join(varargs)
    full_command = f"{command_str} {varargs_str}"
    
    # Check if command length exceeds the platform max length
    if len(full_command) > _platform_max_length:
        raise ValueError(f"Command exceeds platform max length of {_platform_max_length} characters")

    # Adjust target_concurrency for subprocess
    if target_concurrency > 1 and not color:
        raise ValueError("Concurrency > 1 is not supported without color output")

    # Execute the command
    result = subprocess.run(full_command, shell=True, **kwargs)
    
    if result.returncode!= 0:
        raise subprocess.CalledProcessError(result.returncode, full_command)
    
    return result.stdout

# Example usage:
# xargs(("echo", "Hello", "World"), ("one", "two", "three"), color=True)