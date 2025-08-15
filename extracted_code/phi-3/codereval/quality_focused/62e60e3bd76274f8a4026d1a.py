class Bookmarks:
    def __init__(self, bookmarks=None):
        self.bookmarks = bookmarks if bookmarks is not None else []

    @classmethod
    def from_raw_values(cls, values):
        bookmarks = []
        for value in values:
            try:
                bookmarks.append(cls(value))
            except Exception as e:
                # Handle or ignore exception as needed
                pass
        return cls(bookmarks=bookmarks)