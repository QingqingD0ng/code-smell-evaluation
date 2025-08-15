import subprocess

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    if not isinstance(commands, list):
        commands = [commands]
    
    for command in commands:
        full_command = [command] + args
        if verbose:
            print(f"Running: {' '.join(full_command)}")
    
        if hide_stderr:
            stderr = subprocess.DEVNULL
        else:
            stderr = None

        result = subprocess.run(
            full_command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env
        )

        if verbose:
            print(f"Command finished with return code: {result.returncode}")

        return result.stdout, result.returncode