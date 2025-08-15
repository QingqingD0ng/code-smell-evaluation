from textwrap import indent

class PrettyObject:
    def __init__(self, obj):
        self.obj = obj

    def pretty(self, indent=0, debug=False):
        debug_details = ""
        if debug:
            debug_details = ", debug=True"
        obj_repr = "'{}'".format(self.obj) if isinstance(self.obj, str) else repr(self.obj)
        formatted = f"{indent(obj_repr,'' * indent)}{self.__class__.__name__}{debug_details}"
        return formatted

# Example usage:
if __name__ == "__main__":
    obj = PrettyObject("Hello, World!")
    print(obj.pretty(indent=2, debug=True))