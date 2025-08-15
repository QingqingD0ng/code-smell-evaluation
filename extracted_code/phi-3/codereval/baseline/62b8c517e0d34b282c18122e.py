import traceback

class Exttostr:
    @classmethod
    def extostr(cls, e, max_level=30, max_path_level=5):
        tb = traceback.extract_tb(e.__traceback__)
        formatted_traceback = []
        
        for frame in reversed(tb[-max_path_level:]):
            formatted_traceback.append(f"File \"{frame.filename}\", line {frame.lineno}, in {frame.name}")
            if len(formatted_traceback) >= max_path_level:
                break
        
        exception_type = type(e).__name__
        exception_message = str(e)
        formatted_traceback.reverse()
        
        return f"{exception_type}: {exception_message}\n{''.join(formatted_traceback)}"