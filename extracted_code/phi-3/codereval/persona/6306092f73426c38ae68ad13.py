import subprocess

def ansible_playbook(ir_workspace, ir_plugin, playbook_path, verbose=None, extra_vars=None, ansible_args=None):
    command = ['ansible-playbook', playbook_path]
    
    if verbose is not None:
        command.extend(['-v', str(verbose)])
    
    if extra_vars is not None:
        command.extend(['-e', extra_vars])
    
    if ansible_args is not None:
        for arg, value in ansible_args.items():
            command.extend([f"--{arg}", value])
    
    # Assuming ir_workspace and ir_plugin have a method to provide necessary paths and environment variables
    env_vars = ir_workspace.get_env_vars()
    command.extend(env_vars)
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode!= 0:
        raise Exception(f"Ansible Playbook failed: {result.stderr}")
    else:
        print(result.stdout)