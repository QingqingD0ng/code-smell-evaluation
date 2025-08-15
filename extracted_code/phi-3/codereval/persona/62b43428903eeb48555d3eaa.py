import sqlite3
from typing import AnyStr, Union, Iterable, Dict, Sequence, Tuple

class SQLFormatter:
    def __init__(self):
        self._converter = None  # Assume this is initialized elsewhere with the appropriate logic

    def _convert_many(self, sql: str, params: Iterable) -> Tuple[str, Iterable]:
        # This method would use self._converter to convert the SQL and parameters
        # For demonstration purposes, we'll just return the SQL and params as-is
        return sql, params

    def formatmany(self, sql: AnyStr, many_params: Union[Iterable[Dict[Union[str, int], Any]], Iterable[Sequence[Any]]]) -> Tuple[AnyStr, Union[List[Dict[Union[str, int], Any]], List[Sequence[Any]]]]:
        return self._convert_many(sql, many_params)

# Example usage:
if __name__ == "__main__":
    formatter = SQLFormatter()
    sql = "SELECT * FROM users WHERE id IN (?,?,?);"
    params = [1, 2, 3]
    formatted_sql, formatted_params = formatter.formatmany(sql, params)
    print(formatted_sql)  # Output: SELECT * FROM users WHERE id IN (?,?,?);
    print(formatted_params)  # Output: [1, 2, 3]