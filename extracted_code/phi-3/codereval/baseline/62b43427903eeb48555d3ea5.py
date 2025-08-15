from typing import Any, Dict, Sequence, Tuple, Union

class SQLFormatter:
    def __init__(self, converter):
        self._converter = converter

    def format(self, sql: AnyStr, params: Union[Dict[Union[str, int], Any], Sequence[Any]]) -> Tuple[AnyStr, Union[Dict[Union[str, int], Any], Sequence[Any]]]:
        if isinstance(params, dict):
            formatted_sql = self._converter.convert(sql, params)
        else:
            formatted_sql = self._converter.convert(sql, list(params))
        return formatted_sql, params