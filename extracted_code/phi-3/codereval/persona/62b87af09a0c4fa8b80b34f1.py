class Histogram:
    def __init__(self, width, height, left, right, bottom, top):
        self.width = width
        self.height = height
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.data = [[0 for _ in range(width)] for _ in range(height)]

    def fill(self, coord, weight=1):
        x, y = coord
        if self.left <= x <= self.right and self.bottom <= y <= self.top:
            self.data[y - self.bottom][x - self.left] += weight