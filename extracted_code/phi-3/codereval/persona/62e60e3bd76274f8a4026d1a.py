class Bookmarks:
    def __init__(self, bookmarks):
        self.bookmarks = bookmarks

    @classmethod
    def from_raw_values(cls, values):
        return cls(values)

    def __repr__(self):
        return f"Bookmarks({self.bookmarks})"