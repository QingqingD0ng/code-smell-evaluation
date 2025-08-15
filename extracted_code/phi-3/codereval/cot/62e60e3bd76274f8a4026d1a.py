class Bookmarks:
    def __init__(self, bookmarks):
        self.bookmarks = bookmarks

    @classmethod
    def from_raw_values(cls, values):
        return cls(values)

    def __str__(self):
        return f"Bookmarks({self.bookmarks})"

# Example usage is not included as per the instruction to avoid comments.