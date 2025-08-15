class MetaDeterminer:
    @staticmethod
    def determine_metaclass(bases, explicit_mc=None):
        if explicit_mc:
            return explicit_mc
        elif bases:
            base_metaclasses = {base.__class__ for base in bases}
            if len(base_metaclasses) == 1:
                return base_metaclasses.pop()
            else:
                return type
        else:
            return type

metaclass = MetaDeterminer.determine_metaclass((A, B))
print(metaclass)  # Output: <class '__main__.B'>

metaclass = MetaDeterminer.determine_metaclass((A, B), explicit_metaclass)
print(metaclass)  # Output: <class '__main__.type'>

metaclass = MetaDeterminer.determine_metaclass()
print(metaclass)  # Output: <class 'type'>