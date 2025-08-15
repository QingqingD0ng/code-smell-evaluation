def get_parser_option_specs(self, command_name):

    parser = argparse.ArgumentParser()

    commands = {

       'main': {

            'options': [

                {'name': '--verbose', 'action':'store_true', 'help': 'increase verbosity'},

                {'name': '--path', 'type': str, 'help':'specify path for operations'},

            ],

           'required_args': [],

        },

        'virsh': {

            'options': [

                {'name': '--list', 'action':'store_true', 'help': 'list all VMs'},

                {'name': '--create','metavar': 'VMNAME', 'help': 'create a new VM'},

                {'name': '--destroy','metavar': 'VMNAME', 'help': 'destroy an existing VM'},

            ],

           'required_args': ['VMNAME'],

        },

        'ospd': {

            'options': [

                {'name': '--start', 'action':'store_true', 'help':'start the OSPD service'},

                {'name': '--stop', 'action':'store_true', 'help':'stop the OSPD service'},

            ],

           'required_args': [],

        },

    }

    if command_name in commands:

        return commands[command_name]['options']

    else:

        raise ValueError(f"Unknown command: {command_name}")