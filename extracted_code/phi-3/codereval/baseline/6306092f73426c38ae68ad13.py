import subprocess
import os

def ansible_playbook(ir_workspace, ir_plugin, playbook_path, verbose=None, extra_vars=None, ansible_args=None):
    # Set default values if not provided
    if verbose is None:
        verbose = os.getenv('ANSIBLE_VERBOSITY', 'vv')
    if extra_vars is None:
        extra_vars = {}
    if ansible_args is None:
        ansible_args = {}

    # Construct the Ansible command
    command = [
        'ansible-playbook',
        playbook_path,
        '--extra-vars',''.join([f'{key}={value}' for key, value in extra_vars.items()]),
        '--vv' + verbose,
    ] + [f'--{k} {v}' for k, v in ansible_args.items()]

    # Add the Infrared Workspace and InfraredPlugin paths to the command
    command += [
        f'--inventory {ir_workspace.path}',
        f'--limit {ir_plugin.target_group_names}',
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode!= 0:
        raise Exception(f'Ansible playbook failed with return code {result.returncode}: {result.stderr}')

    return result.stdout