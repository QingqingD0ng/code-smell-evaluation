class SubclassesCollector:
    @staticmethod
    def all_subclasses(cls):
        subclasses = set(cls.__subclasses__())
        for subclass in subclasses.copy():
            subclasses.update(SubclassesCollector.all_subclasses(subclass))
        return subclasses