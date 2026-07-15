import random
import math
import ContinuousDistribution

"""
Class for copulas - a building block for the implementation of ContinuousJointDistributions
"""

class Copula:
    def c(self, x : float, y : float) -> float:
        return 0
    def conditional_x(self, x : float, y : float) -> float:
        return 0
    def conditional_y(self, y : float, x : float) -> float:
        return 0
    def inv_conditional_x(self, x : float, y : float) -> float:
        return 0
    def inv_conditional_y(self, y : float, x : float) -> float:
        return 0
    def sample(self) -> tuple[float, float]:
        """
            The general CDF method of sampling from a 2D continuous distribution
        """
        rng_x = random.random()
        rng_y = random.random()

        y_conditioned_on_x = self.inv_conditional_y(rng_y, rng_x)

        return (rng_x, y_conditioned_on_x)
    def sample_with_given_rng(self, rng_x : float, rng_y : float) -> tuple[float, float]:
        y_conditioned_on_x = self.inv_conditional_y(rng_y, rng_x)

        return (rng_x, y_conditioned_on_x)


class IndependenceCopula(Copula):
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        return x * y
    def conditional_x(self, x : float, y : float) -> float:
        return x
    def conditional_y(self, y : float, x : float) -> float:
        return y
    def inv_conditional_x(self, x : float, y : float) -> float:
        return x
    def inv_conditional_y(self, y : float, x : float) -> float:
        return y

    
class Plackett(Copula):
    def __init__(self, theta : float):
        self.theta = theta
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        A = 1 + (self.theta - 1) * (x + y)
        return ((A) - math.sqrt((A) ** 2 - 4 * x * y * self.theta * (self.theta - 1))) / (2 * (self.theta - 1))
    def conditional_x(self, x : float, y : float):
        a = self.theta - 1
        b = 1 + a * (x + y)
        discriminant = b ** 2 - 4 * x * y * self.theta * a
        numerator = math.sqrt(discriminant) + (self.theta + 1) * x - a * y - 1
        denominator = 2 * math.sqrt(discriminant)
        return numerator / denominator
    def inv_conditional_y(self, y : float, x : float) -> float:
        A = y * (1.0 - y)
        B = y * (1.0 - y) * (self.theta - 1.0) + x * (self.theta * y + 1.0 - y)
        C = self.theta * x * y
        
        discriminant = max(0.0, B**2 - 4 * A * C)
        
        # Denominator check to avoid zero division at extreme edges
        denominator = 2 * A * (self.theta - 1.0)
        if math.isclose(denominator, 0.0):
            return y
            
        numerator = B - math.sqrt(discriminant)
        v = numerator / (self.theta - 1.0)
        
        # Ensure floating-point errors don't push v out of [0, 1] bounds
        return max(0.0, min(1.0, v))


class Gaussian(Copula):
    def __init__(self, correlation : float):
        self.correlation = correlation
        self.Normal = ContinuousDistribution.Normal(0,1)
    def bivariate_standard_normal(self, x : float, y : float) -> float:
        a = 1 - self.correlation ** 2
        S = 1 / (2 * math.pi * math.sqrt(a))
        exponent_1 = -1 / (2 * a)
        exponent_2 = x ** 2 - 2 * self.correlation * x * y + y ** 2
        exponent = exponent_1 * exponent_2
        return S * math.e ** exponent
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        return self.bivariate_standard_normal(self.Normal.quantile(x), self.Normal.quantile(y))
    def conditional_x(self, x : float, y : float) -> float:
        numerator = self.Normal.quantile(x) - self.correlation * self.Normal.quantile(y)
        denominator = math.sqrt(1 - self.correlation ** 2)
        return self.Normal.cdf(numerator / denominator)
    def inv_conditional_y(self, y : float, x : float) -> float:
        element1 = self.correlation * self.Normal.quantile(x)
        element2 = math.sqrt(1 - self.correlation ** 2) * self.Normal.quantile(y)
        return self.Normal.cdf(element1 + element2)
    

class Clayton(Copula):
    def __init__(self, theta : float):
        self.theta = theta
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        return max(x ** -self.theta + y ** -self.theta -1, 0) ** (1 / self.theta)
    def conditional_x(self, x: float, y: float) -> float:
        return y ** (-self.theta - 1) * (y ** -self.theta + x ** -self.theta - 1) ** (-1/self.theta - 1)
    def inv_conditional_y(self, y: float, x: float) -> float:
        return (x ** -self.theta * (y ** (-self.theta / (self.theta + 1)) -1) + 1) ** (-1 / self.theta)

class Gumbel(Copula):
    def __init__(self, theta : float):
        self.theta = theta
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        exponent = -(-math.log(x) ** self.theta -math.log(y) ** self.theta) ** (1 / self.theta)
        return math.e ** exponent
    def sample(self) -> tuple[float, float]:
        rng_x = random.random()
        rng_y = random.random()

        rng_u = random.random()
        rng_W = random.random() * math.pi

        a = 1 / self.theta

        V = (math.sin(1 - a) * rng_W / rng_u) ** ((1 - a) / a) * (math.sin(a * rng_W) / math.sin(rng_W)) ** (1 / a)

        exp_x = -(-math.log(rng_x) / V) ** a
        exp_y = -(-math.log(rng_y) / V) ** a

        return (math.e ** exp_x, math.e ** exp_y)
    def sample_with_given_rng(self, rng_x : float, rng_y : float) -> tuple[float, float]:
        rng_u = random.random()
        rng_W = random.random() * math.pi

        a = 1 / self.theta

        V = (math.sin(1 - a) * rng_W / rng_u) ** ((1 - a) / a) * (math.sin(a * rng_W) / math.sin(rng_W)) ** (1 / a)

        exp_x = -(-math.log(rng_x) / V) ** a
        exp_y = -(-math.log(rng_y) / V) ** a

        return (math.e ** exp_x, math.e ** exp_y)

class Frank(Copula):
    def __init__(self, theta : float):
        self.theta = theta
    def c(self, x : float, y : float) -> float:
        if x < 0 or x > 1 or y < 0 or y > 1:
            return None
        numerator = (math.e ** (self.theta * x) -1) * (math.e ** (self.theta * y) -1)
        denominator = math.e ** -self.theta - 1
        return -1 / self.theta * math.log(1 + numerator / denominator)
    def conditional_x(self, x: float, y: float) -> float:
        numerator = math.e ** (-self.theta * y) * (math.e ** (-self.theta * x) - 1)
        denominator = math.e ** -self.theta - 1 + math.e ** (-self.theta * y) * (math.e ** (-self.theta * x) - 1)
        return numerator / denominator
    def inv_conditional_y(self, y: float, x: float) -> float:
        numerator = y * (math.e ** -self.theta - 1)
        denominator = math.e ** -(self.theta * x) + y * (1 - math.e ** -(self.theta * x))
        return -1 / self.theta * math.log(1 + numerator / denominator)