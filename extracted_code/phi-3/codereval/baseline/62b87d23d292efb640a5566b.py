import subprocess

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    if not isinstance(commands, list):
        commands = [commands]
    if not env:
        env = os.environ.copy()
    
    command_str =''.join(commands)
    command_process = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE if verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        shell=True
    )
    
    stdout, stderr = command_process.communicate()
    return command_process.returncode, stdout, stderr