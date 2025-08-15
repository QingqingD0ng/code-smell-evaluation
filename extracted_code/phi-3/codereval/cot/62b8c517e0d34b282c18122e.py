class ExToStr:
    @classmethod
    def format_exception(cls, e, max_level=30, max_path_level=5):
        def _format_traceback(tb, max_level, prefix=''):
            if max_level <= 0:
                return []
            tb_lines = []
            for filename, lineno, name, line in traceback.extract_tb(tb):
                if filename and lineno > max_path_level:
                    tb_lines.append(f"{prefix}{filename}:{lineno}: {name}")
            return tb_lines + _format_traceback(tb.tb_next, max_level-1, prefix+'  ')

        formatted_tb = _format_traceback(e.__traceback__, max_level, '')
        exception_type, exception_value, exception_traceback = e.__class__, e, e.__traceback__
        return f"{exception_type.__name__}: {str(exception_value)} ({', '.join(formatted_tb)})"