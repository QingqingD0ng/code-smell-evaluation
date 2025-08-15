class Bookmarks:
    def __init__(self, bookmarks):
        self.__bookmarks = bookmarks

    @classmethod
    def from_raw_values(cls, values):
        return cls(values)

    def get_bookmarks(self):
        return self.__bookmarks

# Example usage:
# raw_values = ['Home', 'Work', 'Social Media', 'Shopping']
# bookmarks_obj = Bookmarks.from_raw_values(raw_values)
# print(bookmarks_obj.get_bookmarks())