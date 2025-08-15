import csv
from typing import Iterable, Optional

def to_csv(self, separator=",", header=None):
    csv_data = []
    if header is not None:
        csv_data.append(header)
    
    for point in self.graph_points:
        row = separator.join(str(value) for value in point)
        csv_data.append(row)
    
    return "\n".join(csv_data)