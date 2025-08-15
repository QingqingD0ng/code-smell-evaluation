import subprocess
import json
from typing import List, Dict

class InfraredWorkspace:
    def __init__(self, inventory_file: str, host_group: str):
        self.inventory_file = inventory_file
        self.host_group = host_group

class InfraredPlugin:
    def __init__(self, playbook_path: str):
        self.playbook_path = playbook_path

def _run_playbook(cli_args: List[str], vars_dict: Dict, workspace: InfraredWorkspace, plugin: InfraredPlugin) -> str:
    vars_json = json.dumps(vars_dict)
    command = ["ansible-playbook"] + cli_args + [
        "-e", vars_json,
        "--extra-vars", vars_json,
        "--inventory", workspace.inventory_file,
        "--limit", workspace.host_group,
        plugin.playbook_path
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode!= 0:
        raise Exception(f"Ansible Playbook failed: {process.stderr}")
    return process.stdout

# Example usage:
# vars = {'key': 'value'}
# cli_args = ['-i', 'localhost,127.0.0.1']
# workspace = InfraredWorkspace('inventory.ini', 'webservers')
# plugin = InfraredPlugin('site.yml')
# results = _run_playbook(cli_args, vars, workspace)