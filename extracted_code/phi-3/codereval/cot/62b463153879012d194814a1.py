class MediaRenditionManager:
    def __init__(self):
        self._renditions = {}

    def add_rendition(self, lang, file_path):
        self._renditions[lang] = file_path

    def file_path(self, lang):
        return self._renditions.get(lang)

manager = MediaRenditionManager()
manager.add_rendition('en', 'path/to/english_rendition.mp4')
print(manager.file_path('en'))  # Outputs: path/to/english_rendition.mp4