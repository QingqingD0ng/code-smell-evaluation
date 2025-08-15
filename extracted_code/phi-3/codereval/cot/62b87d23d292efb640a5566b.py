import subprocess
import shlex

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    if isinstance(commands, str):
        commands = [commands]
    if isinstance(args, str):
        args = [args]
    command_args = []
    for cmd in commands:
        command_args.extend(shlex.split(cmd))
    command_args.extend(args)
    
    env = env or {}
    if verbose:
        print(f"Running command: {' '.join(command_args)}")
    
    stderr_option = subprocess.DEVNULL if hide_stderr else None
    result = subprocess.run(
        command_args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=stderr_option,
        text=True
    )
    
    if verbose and result.returncode!= 0:
        print(f"Command failed with return code {result.returncode}")
    
    return result.stdout.strip() if hide_stderr else result