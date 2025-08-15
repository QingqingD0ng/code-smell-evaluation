def extend_cli(self, root_subparsers):

    spec_parser = root_subparsers.add_parser('spec', help='Handle spec-related tasks')

    spec_parser.add_argument('--version', type=str, help='Spec version to use')

    spec_parser.add_argument('--config', type=str, help='Path to configuration file')