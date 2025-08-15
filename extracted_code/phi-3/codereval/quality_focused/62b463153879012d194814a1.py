class RenditionManager:
    def __init__(self):
        self._renditions = {}

    def add_rendition(self, lang, file_path):
        self._renditions[lang] = file_path

    def get_rendition(self, lang):
        return self._renditions.get(lang, None)

# Usage example
rendition_manager = RenditionManager()
rendition_manager.add_rendition('en', '/path/to/english/file.txt')
rendition_manager.add_rendition('es', '/path/to/spanish/file.txt')

print(rendition_manager.get_rendition('en'))  # Output: /path/to/english/file.txt
print(rendition_manager.get_rendition('es'))  # Output: /path/to/spanish/file.txt
print(rendition_manager.get_rendition('fr'))  # Output: None