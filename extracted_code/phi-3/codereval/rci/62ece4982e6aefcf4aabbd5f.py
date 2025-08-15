from typing import List, Generator

def paging(response: List[any], max_results: int) -> Generator[List[any], None, None]:
    if not isinstance(response, list):
        raise ValueError("Invalid input: response must be a list.")
    if not isinstance(max_results, int):
        raise ValueError("Invalid input: max_results must be an integer.")
    if max_results <= 0:
        raise ValueError("Invalid input: max_results must be greater than 0.")
    if not response:
        return
    for i in range(0, len(response), max_results):
        yield response[i:i + max_results]