import subprocess
from ansible.parsing.dataloader import DataLoader
from ansible.vars.manager import VariableManager
from ansible.inventory.manager import InventoryManager
from ansible.playbook.play import Play
from ansible.executor.task_queue_manager import TaskQueueManager

def _run_playbook(cli_args, vars_dict, ir_workspace, ir_plugin):
    loader = DataLoader()
    variable_manager = VariableManager(loader=loader)
    variable_manager.extra_vars = vars_dict
    inventory = InventoryManager(loader=loader, sources=ir_workspace.inventory_sources)

    play = Play().include_vars(loader=loader, file_name=ir_workspace.ansible_vars_file)
    play.hosts = inventory.get_hosts(ir_workspace.target_group)
    playbook = [play]

    tqm = TaskQueueManager(
        inventory=inventory,
        variable_manager=variable_manager,
        loader=loader,
        passwords=ir_workspace.passwords,
        stdout_callback='yaml'
    )

    # Add the plugin to the playbook if it's not already included
    if ir_plugin not in playbook[0].block_collections:
        playbook[0].block_collections['all'] = [ir_plugin]

    result = subprocess.run(
        ['ansible-playbook'] + cli_args + ['-i', ir_workspace.inventory_file, '-e', str(vars_dict)],
        capture_output=True,
        text=True,
        check=True,
        env=tqm.env
    )

    return result.stdout