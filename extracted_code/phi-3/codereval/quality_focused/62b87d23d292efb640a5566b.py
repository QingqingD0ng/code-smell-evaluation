import subprocess

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    if isinstance(commands, str):
        commands = [commands]
    
    for command in commands:
        args_list = [command] + args
        if env:
            env = dict(os.environ, **env)
        
        stdout_option = subprocess.PIPE if verbose or not hide_stderr else None
        stderr_option = subprocess.STDOUT if verbose or not hide_stderr else None
        
        process = subprocess.run(args_list, cwd=cwd, env=env, stdout=stdout_option, stderr=stderr_option, check=True)
        
        if verbose:
            print(process.stdout.decode())
        if not hide_stderr:
            print(process.stderr.decode())
        
        return process.returncode