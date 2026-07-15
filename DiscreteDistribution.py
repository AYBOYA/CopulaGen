import random
import math
import numpy as np

"""
    Parent class of discrete probability distributions
"""

class DiscreteDistribution:
    def __init__(self):
        self.name = "Unspecified Distribution"
    
    def evaluate_pdf(self, value : int) -> float:
        """
            Returns probability of a random variable with distribution being equal to value

            =====================================================
            SHOULD BE OVERRIDDEN BY EVERY CONCRETE IMPLEMENTATION
            =====================================================

            args:
                value - integer to evaluate distribution at
            
            returns:
                probability of a random variable with the distribution having the given value
        """
        return 0

    def evaluate_cdf(self, value : int) -> float:
        """
            Returns probability of a random variable with distribution being less than or equal to value

            =====================================================
            THIS IMPLEMENTATION ASSUMES THAT SMALLEST VALUE WITH POSITIVE PROBABILITY IS >= 0

            SHOULD BE OVERRIDDEN BY EVERY CONCRETE IMPLEMENTATION IN WHICH THIS IS NOT TRUE
            =====================================================

            args:
                value - integer to evaluate distribution at
            
            returns:
                probability of a random variable with the distribution being less than or equal to the given value
        """
        summation = 0

        for i in range(value + 1):
            summation += self.evaluate_pdf(i)
        
        return summation
    
    def sample(self) -> int:
        """
            Returns a single value based on the distribution
        """
        value = 0
        rand = random.random()
        
        cdf_val = self.evaluate_cdf(value)

        if rand > cdf_val:
            while rand > cdf_val:
                value += 1
                cdf_val = self.evaluate_cdf(value)
            return value
        else:
            while rand <= cdf_val:
                value -= 1
                cdf_val = self.evaluate_cdf(value)
            return value + 1
    
    def gen_samples(self, size : int) -> np.array:
        """
            Return np array of size of independent samples based on the distribution
        """
        return np.transpose(np.array([[self.sample() for i in range(size)]]))


class Uniform(DiscreteDistribution):
    def __init__(self, low : int, high : int):
        self.name = "Uniform"
        self.low = low
        self.high = high
    
    def evaluate_pdf(self, value: int) -> float:
        if value >= self.low and value <= self.high:
            return 1 / (self.high - self.low + 1)
        return 0
    
    def evaluate_cdf(self, value: int) -> float:
        if value < self.low:
            return 0
        elif value <= self.high:
            difference = abs(self.high - self.low) + 1
            return (abs(value - self.low) + 1) / difference
        return 1
    

class Bernoulli(DiscreteDistribution):
    def __init__(self, p : float):
        self.name = "Bernoulli"
        self.p = p
    
    def evaluate_pdf(self, value: int) -> float:
        """
            Equation: p^x * (1-p)^(1-x)
        """
        if value == 0 or value == 1:
            return (self.p ** value) * ((1 - self.p) ** (1 - value))
        return 0

class Binomial(DiscreteDistribution):
    def __init__(self, n : int, p : float):
        self.name = "Binomial"
        self.n = n
        self.p = p
    
    def evaluate_pdf(self, value: int) -> float:
        if value < 0:
            return 0
        return math.comb(self.n,value) * (self.p ** value) * ((1 - self.p) ** (self.n - value))

class Geometric(DiscreteDistribution):
    def __init__(self, p : float):
        self.name = "Geometric"
        self.p = p
    
    def evaluate_pdf(self, value: int) -> float:
        if value < 1:
            return 0
        return ((1 - self.p) ** (value - 1)) * self.p

b = Geometric(0.4)
print(b.gen_samples(10))
print(b.evaluate_pdf(2))