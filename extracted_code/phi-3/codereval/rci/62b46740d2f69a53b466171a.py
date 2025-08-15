class PrettyObject:
    def __init__(self, obj):
        self.obj = obj

    def pretty(self, indent=0, debug=False):
        debug_details = ", debug=True" if debug else ""
        if isinstance(self.obj, str):
            obj_representation = f"'{self.obj}'"
        else:
            obj_representation = repr(self.obj)
        return "\n".join([" " * indent + line for line in obj_representation.split("\n")]) + debug_details