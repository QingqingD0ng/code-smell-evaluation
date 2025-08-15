import csv

class Graph:
    # Assuming the Graph class has a method to iterate over points
    # and each point has a 'coordinates' attribute and a 'values' attribute.
    def __iter__(self):
        # This should be the actual implementation
        pass

    def to_csv(self, separator=",", header=None):
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=separator)
            if header:
                writer.writerow(header)
            for point in self:
                row = [str(coord) for coord in point.coordinates]
                row.extend(str(value) for value in point.values)
                writer.writerow(row)