import yaml

def _dump_string(obj):
    return yaml.dump(obj, allow_unicode=True)