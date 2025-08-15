import subprocess

def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False, env=None):
    result = subprocess.run(commands + args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if not hide_stderr else subprocess.DEVNULL, env=env)
    return result.stdout, result.returncode