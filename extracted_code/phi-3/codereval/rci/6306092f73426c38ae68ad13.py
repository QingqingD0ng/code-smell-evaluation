import subprocess

def run_command(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode!= 0:
        raise Exception(f'Command failed with return code {res.returncode}: {res.stderr}')
    return res.stdout

def ansible_playbook(ir_workspace, ir_plugin, playbook_path, verbose=None, extra_vars=None, ansible_args=None):
    if verbose is None:
        verbose = os.getenv('ANSIBLE_VERBOSITY', 'vv')
    if extra_vars is None:
        extra_vars = {}
    if ansible_args is None:
        ansible_args = {}

    cmd = ['ansible-playbook', playbook_path]
    cmd.extend(['--extra-vars', f'{k}={v}' for k, v in extra_vars.items()])
    cmd.append('--vv' + verbose)
    cmd.extend([
        f'--inventory {ir_workspace.path}',
        f'--limit {ir_plugin.target_group_names}',
    ])
    cmd.extend([f'--{k} {v}' for k, v in ansible_args.items()])

    output = run_command(cmd)
    return output