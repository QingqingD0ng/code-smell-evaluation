from typing import Iterable, Optional


class Graph:
    # Graph implementation remains as is
    def get_points(self) -> Iterable[Iterable]:
        pass


def to_csv(graph: Graph, separator: str = ",", header: Optional[str] = None) -> str:
    def convert_point(point: Iterable) -> str:
        return separator.join(str(part) for part in point)

    csv_lines = []

    if header is not None:
        csv_lines.append(header)

    for point in graph.get_points():
        csv_lines.append(convert_point(point))

    return "\n".join(csv_lines)