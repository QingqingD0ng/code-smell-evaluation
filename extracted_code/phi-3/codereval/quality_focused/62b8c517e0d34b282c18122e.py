class ExToStrFormatter:
    @classmethod
    def extostr(cls, e, max_level=30, max_path_level=5):
        def format_traceback(exc_traceback, max_level, max_path_level):
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            formatted_trace = []
            for line in tb_lines:
                if 'File' in line and ':' in line:
                    file_info = line.split(':', 1)[0]
                    if 'File' in file_info:
                        path_parts = file_info.split()
                        if len(path_parts) > max_path_level:
                            path_info = '...' +''.join(path_parts[-(max_path_level-3):])
                        else:
                            path_info =''.join(path_parts)
                        line = f"{path_info}: {line.split('File', 1)[1]}"
                formatted_trace.append(line)
                if len(formatted_trace) > max_level:
                    break
            return formatted_trace

        tb = traceback.format_exception(type(e), e, e.__traceback__)
        return ''.join(format_traceback(e.__traceback__, max_level, max_path_level))

# Usage example:
try:
    1 / 0
except Exception as e:
    print(ExToStrFormatter.extostr(e))