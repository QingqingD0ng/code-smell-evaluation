class FileSplitter:

    def __init__(self):

        self._before = None

        self._after = None


    def set_cut_chars(self, before: bytes, after: bytes) -> None:

        if not isinstance(before, bytes):

            raise TypeError("'before' must be of type 'bytes'")

        if not isinstance(after, bytes):

            raise TypeError("'after' must be of type 'bytes'")


        self._before = before

        self._after = after