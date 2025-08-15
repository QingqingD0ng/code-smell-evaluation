class SpecResolver:
    def _include_groups(self, parser_dict):
        included_groups = {}
        for group_name, file_paths in parser_dict.items():
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    content = file.read()
                    included_groups.update(self._parse_include_directives(content, group_name))
        return included_groups

    def _parse_include_directives(self, content, group_name):
        include_groups = {}
        #... parse the content to find include directives and populate include_groups...
        return include_groups

# Example usage:
# resolver = SpecResolver()
# included_groups = resolver._include_groups(parser_dict)