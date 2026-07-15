import random
import math
import Distribution

"""
    Parent class of continuous probability distributions
"""

class ContinuousDistribution(Distribution.Distribution):
    def pdf(self, value : float) -> float:
        return 0
    def cdf(self, value : float) -> float:
        return 0
    def ccdf(self, value : float) -> float:
        return 1 - self.cdf(value)
    def quantile(self, value : float) -> float:
        return 0
    def sample(self) -> float:
        rng = random.random()
        return self.quantile(rng)

class ContinuousUniform(ContinuousDistribution):
    def __init__(self, a : float, b : float):
        self.a = min(a,b)
        self.b = max(a,b)
        self.difference = self.b - self.a
    
    def pdf(self, value : float) -> float:
        if value < self.a or value > self.b:
            return 0
        return 1.0 / self.difference
    def cdf(self, value : float) -> float:
        if value < self.a:
            return 0
        if value > self.b:
            return 1
        return (value - self.a) / (self.difference)
    def quantile(self, value : float) -> float:
        if value < 0 or value > 1:
            return None
        return value * self.difference + self.a

class Exponential(ContinuousDistribution):
    def __init__(self, l : float):
        self.l = l
    
    def pdf(self, value : float) -> float:
        if value < 0:
            return 0
        return self.l * math.e ** (-self.l * value)
    def cdf(self, value : float) -> float:
        if value < 0:
            return 0
        return 1 - math.e ** (-self.l * value)
    def quantile(self, value : float) -> float:
        if value < 0 or value > 1:
            return None
        return -math.log(1 - value)/self.l

class Normal(ContinuousDistribution):
    def __init__(self, mean : float, std : float):
        self.mean = mean
        self.std = std
    
    def pdf(self, value : float) -> float:
        if value < 0:
            return 0
        return math.e ** ((-(value - self.mean) ** 2) / (2 * self.std ** 2)) / (self.std * math.sqrt(2 * math.pi))
    def cdf(self, value : float) -> float:
        #https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1007&context=emse_fac_pubs
        #Bowling-Khasawneh-Kaewkuekool-Cho Logistic Approximation (2009)
        standardized = (value - self.mean) / self.std
        return 1 / (1 + math.e ** -(0.07056 * standardized ** (3) + 1.5976 * standardized))
    def quantile(self, value : float) -> float:
        if value < 0 or value > 1:
            return None
        # 1. Compute target logit
        C = math.log(value / (1 - value))
        
        # 2. Solve the cubic equation for z (Cardano's method)
        A, B = 0.07056, 1.5976
        p = B / A
        q = -C / A
        
        discriminant = (q / 2)**2 + (p / 3)**3
        sqrt_disc = math.sqrt(discriminant)
        
        z = math.cbrt(-q / 2 + sqrt_disc) + math.cbrt(-q / 2 - sqrt_disc)
        
        # 3. Rescale z back to x
        return self.mean + z * self.std
