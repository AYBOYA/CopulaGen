import Distribution
import ContinuousDistribution
import Copula

"""
Class for joint probability distribution for continuous marginals
"""

class ContinuousJointDistribution(Distribution.Distribution):
    def __init__(self, c : Copula.Copula, x : ContinuousDistribution.ContinuousDistribution, y : ContinuousDistribution.ContinuousDistribution):
        self.c = c
        self.x = x
        self.y = y
    
    def sample(self, marginal_transform = True) -> tuple[float, float]:
        copula_sample = self.c.sample()

        copula_x = copula_sample[0]
        copula_y = copula_sample[1]

        if marginal_transform:
            scaled_x = self.x.quantile(copula_x)
            scaled_y = self.y.quantile(copula_y)
            return (scaled_x, scaled_y)

        return (copula_x, copula_y)
    
    def sample_with_given_rng(self, rng_x : float, rng_y : float, marginal_transform = True) -> tuple[float, float]:
        copula_sample = self.c.sample_with_given_rng(rng_x, rng_y)

        copula_x = copula_sample[0]
        copula_y = copula_sample[1]

        if marginal_transform:
            scaled_x = self.x.quantile(copula_x)
            scaled_y = self.y.quantile(copula_y)
            return (scaled_x, scaled_y)

        return (copula_x, copula_y)