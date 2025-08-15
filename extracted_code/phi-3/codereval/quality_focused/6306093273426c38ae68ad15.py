import subprocess
from ansible_runner import run

def _run_playbook(cli_args, vars_dict, ir_workspace, ir_plugin):
    runner = run(
        private_data_dir=ir_workspace,
        inventory=ir_workspace.inventory,
        playbook=ir_workspace.playbook,
        extra_vars=vars_dict,
        runner_options={
            'vault_password': ir_plugin.vault_password,
            'extra_vars': vars_dict
        }
    )
    return runner.json