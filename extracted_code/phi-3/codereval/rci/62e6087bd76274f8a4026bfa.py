class DataContainer:
    def __init__(self, data=[0, 0]):
        self.data = data
    def pop_u16(self):
        if len(self.data) >= 2:
            return (self.data[-2] << 8) | self.data[-1]
        else:
            raise IndexError("pop_u16: too few elements in data")