class DehydratedPoint:
    def __init__(self, value):
        self.value = value
        self.is_dehydrated = self._dehydrate(value)

    def _dehydrate(self, value):
        if len(value) == 1:
            return f"Point-{value}"
        elif len(value) == 2:
            return f"Point[{value[0]}, {value[1]}]"
        elif len(value) == 3:
            return f"Point[{value[0]}, {value[1]}, {value[2]}]"
        else:
            return f"Point[{', '.join(map(str, value))}]"

    def __str__(self):
        return self.is_dehydrated

# Example usage:
point = DehydratedPoint([1, 2, 3])
print(point)  # Output: Point[1, 2, 3]

point_with_one_coordinate = DehydratedPoint(42)
print(point_with_one_coordinate)  # Output: Point-42

point_with_two_coordinates = DehydratedPoint([1, 2])
print(point_with_two_coordinates)  # Output: Point[1, 2]

point_with_three_coordinates = DehydratedPoint([1, 2, 3])
print(point_with_three_coordinates)  # Output: Point[1, 2, 3]