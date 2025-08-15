class FileGrouper:
    def _group_files_by_xml_filename(self, source, xmls, files):
        grouped_files = {}
        for xml in xmls:
            grouped_files[xml] = []
            for file in files:
                if xml in file:
                    grouped_files[xml].append(file)
        return grouped_files