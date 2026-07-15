from typing import Any

"""
Point class to associate label with coordinates
"""

class Point:
    def __init__(self, values : tuple, label : Any = None):
        self.values = values
        self.label = -1
        self.dimensions = len(values)
    
    def to_tuple(self) -> tuple:
        return tuple(self.values[i] for i in range(self.dimensions))

"""
Methods to make lists of data from lists of points
"""

def points_to_tuples(points : list[Point]) -> list[tuple]:
    """
        Takes list of points, returns list of coordinates
    """
    return [point.to_tuple() for point in points]

def points_labels(points : list[Point]) -> list[Any]:
    """
        Takes list of points, returns list of labels of points
    """
    return [point.label for point in points]

def split_tuples(tuples : list[tuple]) -> list[list[float]]:
    """
        Takes list of coordinates, returns list of sets of coordinate values by dimension
    """
    dimension = len(tuples[0])
    num_points = len(tuples)
    split = [[tuples[i][z] for i in range(num_points)] for z in range(dimension)]
    return split