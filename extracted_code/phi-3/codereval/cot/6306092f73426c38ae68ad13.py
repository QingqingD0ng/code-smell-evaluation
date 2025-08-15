import subprocess
import json

class AnsiblePlaybookError(Exception):
    pass

def ansible_playbook(ir_workspace, ir_plugin, playbook_path, verbose=None, extra_vars=None, ansible_args=None):
    if extra_vars is None:
        extra_vars = {}
    if ansible_args is None:
        ansible_args = {}

    workspace_path = ir_workspace.get_path()
    plugin_name = ir_plugin.get_name()

    try:
        command = ['ansible-playbook', playbook_path]
        if verbose is not None:
            command.append(f'--verbose={verbose}')
        if extra_vars:
            command.extend(['--extra-vars', json.dumps(extra_vars)])
        if ansible_args:
            for arg, value in ansible_args.items():
                command.extend([f'--{arg}', value])

        command.append(workspace_path)

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise AnsiblePlaybookError(f'Ansible playbook failed: {e.stderr}')