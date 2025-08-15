import traceback

class ExceptionFormatter:
    @classmethod
    def extostr(cls, e, max_level=30, max_path_level=5):
        formatted_exception = ""
        exception_traceback = traceback.format_exception(type(e), e, e.__traceback__)
        formatted_exception += ''.join(exception_traceback)
        shortened_traceback = "".join(formatted_exception.splitlines()[max_level:])
        
        # Filter out the levels of traceback exceeding max_path_level
        filtered_traceback = "\n".join(traceback.format_tb(e.__traceback__)[:max_path_level])
        
        return f"{filtered_traceback}\n\nCaught exception:\n{shortened_traceback}"

# Example usage:
try:
    raise ValueError("An example error")
except Exception as e:
    print(ExceptionFormatter.extoStr(e))