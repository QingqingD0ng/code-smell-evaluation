import yaml

def _dump_string(obj, dumper=yaml.Dumper):
    return yaml.dump(obj, Dumper=dumper)