import subprocess
import json

def _run_playbook(cli_args, vars_dict, ir_workspace, ir_plugin):
    extra_vars = json.dumps(vars_dict)
    ansible_cmd = ['ansible-playbook', '-e', extra_vars] + cli_args
    process = subprocess.run(ansible_cmd, cwd=ir_workspace.path)
    return process.returncode, process.stdout, process.stderr