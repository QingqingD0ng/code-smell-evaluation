import traceback
from typing import Any

class Exttostr:
    @classmethod
    def extostr(cls, e: Exception, max_level: int = 30, max_path_level: int = 5) -> str:
        tb = traceback.extract_tb(e.__traceback__)
        formatted_traceback = [f"File \"{frame.filename}\", line {frame.lineno}, in {frame.name}" for frame in reversed(tb[-max_path_level:])]
        
        exception_type = type(e).__name__
        exception_message = str(e)
        
        return f"{exception_type}: {exception_message}\n{''.join(formatted_traceback)}"