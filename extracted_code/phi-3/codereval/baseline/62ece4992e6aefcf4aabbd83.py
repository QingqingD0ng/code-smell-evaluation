import subprocess

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    results = []
    for command in commands:
        full_command = [command] + args
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
            if verbose:
                print(f"Command '{command}' failed with return code {e.returncode}")
                if e.stderr:
                    print(f"Error output:\n{e.stderr}")
            results.append((None, e.returncode))
    return results