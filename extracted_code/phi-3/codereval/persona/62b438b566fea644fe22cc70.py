def bash_completion():
    completions = {
        'borg create': '--archives --compression --encryption --label --remote',
        'borg list': '--keydir --keyfile --remote',
        'borg status': '--keydir --keyfile --remote',
        'borg commit': '--keydir --keyfile --remote',
        'borg remove': '--keydir --keyfile --remote',
        'borg rm': '--keydir --keyfile --remote',
        'borg sync': '--keydir --keyfile --remote',
        'borg branch': '--keydir --keyfile --remote',
        'borg clone': '--keydir --keyfile --remote',
        'borg push': '--keydir --keyfile --remote',
        'borg pull': '--keydir --keyfile --remote',
        'borg init': '--keydir --keyfile',
        'borg mount': '--keydir --keyfile',
        'borg unmount': '--keydir --keyfile',
        'borg info': '--keydir --keyfile',
        'borg delete': '--keydir --keyfile',
    }
    return completions