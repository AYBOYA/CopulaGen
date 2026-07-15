import random
import numpy as np
import Distribution
import Point

"""
Class to generate and store data
"""

class DataGenerator:
    def __init__(self, dimensions : int, distribution : Distribution.Distribution, noise : list[float] = None):
        self.dimensions = dimensions
        self.distribution = distribution
        self.points = []
        self.noise = noise
    
    def generate_noise(self, values : tuple) -> list:
        noise_coefficients = [random.random() - 0.5 for i in range(self.dimensions)]
        noise_values = np.multiply(noise_coefficients, self.noise)
        noisified_values = np.add(values, noise_values)
        return noisified_values

    def generate_points(self, num_points : int) -> None:
        new_point_values = [self.distribution.sample() for i in range(num_points)]
        if self.noise:
            new_point_values = [self.generate_noise(new_point_values[i]) for i in range(num_points)]
        new_points = [Point.Point((tuple(new_point_values[i][z] for z in range(self.dimensions)))) for i in range(num_points)]
        self.points = self.points + new_points
    
    def classify(self, index_of_classifier : int, number_of_classes : int, split : list[float]) -> None:
        """
            Give label to points by using their coordinates on a particular dimension and a set of cutoffs
        """
        maximum = max([point.values[index_of_classifier] for point in self.points])
        split = [split[i] * maximum for i in range(len(split))]
        for i in range(number_of_classes):
            cutoff = split[i]
            for z in range(len(self.points)):
                point = self.points[z]
                if point.label == -1 and point.values[index_of_classifier] <= cutoff:
                    point.label = i