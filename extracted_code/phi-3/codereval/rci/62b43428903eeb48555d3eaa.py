from typing import AnyStr, Union, Iterable, Tuple, Dict, Sequence

from collections.abc import IterableABC


class YourClass:
    def __init__(self):
        self._converter = None  # Placeholder for the actual converter logic

    def convert_many_params(self, sql: AnyStr, many_params: Union[Iterable[Dict[Union[str, int], Any]], Iterable[Sequence[Any]]]) -> Tuple[AnyStr, Union[List[Dict[Union[str, int], Any]], List[Sequence[Any]]]]:
        converted_params = [self._converter.convert_many(param) for param in many_params]
        return sql, converted_params