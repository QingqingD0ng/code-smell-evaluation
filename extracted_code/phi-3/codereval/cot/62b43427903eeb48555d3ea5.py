import typing
from typing import Any, Dict, Union, Tuple, Sequence

class SQLFormatter:
    def __init__(self, converter):
        self._converter = converter

    def format(self, sql: Union[str, bytes], params: Union[dict, Sequence]) -> Tuple[Union[str, bytes], Union[dict, Sequence]]:
        converted_sql, converted_params = self._converter.convert(sql, params)
        return converted_sql, converted_params