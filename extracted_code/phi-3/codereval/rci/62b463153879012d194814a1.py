class YourClass:

    def __init__(self):

        self._renditions = {}


    def add_rendition(self, lang: str, file_path: str) -> None:

        if not isinstance(lang, str) or not isinstance(file_path, str):

            raise TypeError("Language and file path must be strings.")

        self._renditions[lang] = file_path


    def get_filepath(self, lang: str) -> str:

        if lang not in self._renditions:

            raise KeyError(f"No rendition found for language: {lang}")

        return self._renditions[lang]


    def default_filepath(self) -> str:

        return self.get_filepath('default')