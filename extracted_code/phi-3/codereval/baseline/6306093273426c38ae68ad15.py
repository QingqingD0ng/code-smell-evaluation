import subprocess
import json

def _run_playbook(cli_args, vars_dict, ir_workspace, ir_plugin):
    # Convert vars_dict to a JSON string to pass as a command-line argument
    vars_json = json.dumps(vars_dict)
    
    # Construct the ansible-playbook command with the required arguments
    command = ["ansible-playbook"] + cli_args + [
        f"-e", f"{vars_json}",
        "--extra-vars", vars_json,
        "--inventory", ir_workspace.inventory_file,
        "--limit", ir_workspace.host_group,
        ir_plugin.playbook_path
    ]
    
    # Run the command and capture the output
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command was successful
    if process.returncode!= 0:
        raise Exception(f"Ansible Playbook failed: {process.stderr}")
    
    # Return the output from the playbook
    return process.stdout