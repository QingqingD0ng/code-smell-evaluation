class DehydratedPoint:
    def __init__(self, value):
        self.value = value
        self.structure = self.determine_structure()

    def determine_structure(self):
        if len(self.value) == 1:
            return'single'
        elif len(self.value) == 2:
            return 'pair'
        elif len(self.value) == 3:
            return 'triple'
        else:
            return 'complex'

    def __str__(self):
        return f"DehydratedPoint with structure '{self.structure}' and value '{self.value}'"