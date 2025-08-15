class ToCSVConverter:
    def to_csv(self, separator=",", header=None):
        converted_rows = []
        for point in self.graph_points:
            coordinates = separator.join(str(part) for part in point.coordinates)
            converted_rows.append(f"{coordinates}{separator}" if header is None else f"{header}{separator}{coordinates}")
        return "\n".join(converted_rows)