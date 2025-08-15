import yaml

def _dump_string(obj, dumper=None):
    if dumper is None:
        dumper = yaml.Dumper
    return yaml.dump(obj, Dumper=dumper)