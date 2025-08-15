class Bookmarks:
    def __init__(self, bookmarks):
        self.bookmarks = bookmarks

    @classmethod
    def from_raw_values(cls, values):
        return cls(values)

# Example usage:
# Assuming Bookmarks is the class where the above method is defined
# raw_values = ['Home', 'Work', 'Social Media', 'Shopping']
# bookmarks_obj = Bookmarks.from_raw_values(raw_values)