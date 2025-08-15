import subprocess
import shlex
import os

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    if not isinstance(commands, list):
        commands = [commands]
    if not env:
        env = os.environ.copy()
    
    command =''.join(shlex.quote(c) for c in commands)
    args = [shlex.quote(arg) for arg in args]
    
    if verbose:
        command = f'echo "{command}"'
    
    command_process = subprocess.Popen(
        args=[command] + args,
        stdout=subprocess.PIPE if verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        text=True,
        shell=True
    )
    
    try:
        stdout, stderr = command_process.communicate()
        return {
           'return_code': command_process.returncode,
           'stdout': stdout,
           'stderr': stderr,
            'command': command
        }
    except Exception as e:
        return {
           'return_code': None,
           'stdout': None,
           'stderr': str(e)
        }