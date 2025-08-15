import typing
from typing import AnyStr, Dict, Sequence, Tuple, Union

class DatabaseFormatter:
    def __init__(self, converter):
        self._converter = converter

    def format(self, sql: AnyStr, params: Union[Dict[Union[str, int], Any], Sequence[Any]]) -> Tuple[AnyStr, Union[Dict[Union[str, int], Any], Sequence[Any]]]:
        converted_sql, converted_params = self._converter.convert(sql, params)
        return converted_sql, converted_params