class MyClass:
    def names_and_descriptions(self, all=False):
        if all:
            for attr_name, attr_value in self.__dict__.items():
                if not attr_name.startswith('_'):
                    print(f"{attr_name}: {attr_value}")
        else:
            print(self.__class__.__name__, getattr(self, self.__class__.__name__.lower(), None))