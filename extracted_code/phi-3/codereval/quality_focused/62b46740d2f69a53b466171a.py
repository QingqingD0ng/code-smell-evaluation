class MyClass:
    def __init__(self, obj):
        self.obj = obj

    def pretty(self, indent=0, debug=False):
        indent_str ='' * indent
        obj_repr = repr(self.obj) if not isinstance(self.obj, str) else f"'{self.obj}'"
        debug_details = f", debug={debug}" if debug else ""
        return f"{indent_str}{self.__class__.__name__}({obj_repr}{debug_details})"