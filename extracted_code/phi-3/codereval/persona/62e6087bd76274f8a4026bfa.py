def pop_u16(self):
    if len(self.data) >= 2:
        return (self.data.pop() << 8) | self.data.pop()