class Histogram:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [[0 for _ in range(width)] for _ in range(height)]

    def fill(self, coord, weight=1):
        x, y = coord
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y][x] += weight

    def __str__(self):
        return '\n'.join(' '.join(str(cell) for cell in row) for row in self.data)

# Example usage:
hist = Histogram(10, 5)
hist.fill((3, 2), weight=5)
print(hist)