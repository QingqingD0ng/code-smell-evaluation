class YourClass:
    def __init__(self):
        self.data = []

    def pop_u16(self):
        if len(self.data) >= 2:
            return (self.data.pop(), self.data.pop())
        else:
            raise IndexError("Not enough elements to pop.")