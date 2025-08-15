class QualityExpert:
    def __init__(self):
        self._renditions = {}

    def add_rendition(self, lang, file_path):
        self._renditions[lang] = file_path

    def filepath(self):
        return self._renditions.get('default', '')