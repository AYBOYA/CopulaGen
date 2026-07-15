import random
import Distribution
import ContinuousJointDistribution

class DVine_3(Distribution.Distribution): #Note: not really a distribution - just able to sample from it so for these purposes it acts as if it were a distribution
    def __init__(self, j1 : ContinuousJointDistribution.ContinuousJointDistribution, j2 : ContinuousJointDistribution.ContinuousJointDistribution, j3 : ContinuousJointDistribution.ContinuousJointDistribution):
        self.joints = [j1, j2, j3]
        self.marginals = [j1.x, j1.y, j2.y]
    
    def sample(self) -> tuple[float, float, float]:
        rng_x = random.random()
        rng_y = random.random()
        rng_z = random.random()
        generated_y = self.joints[0].sample_with_given_rng(rng_x, rng_y, marginal_transform = False)[1]
        
        conditioning_input = self.joints[0].c.conditional_x(rng_x, generated_y)
        w = self.joints[2].c.inv_conditional_y(rng_z, conditioning_input)

        generated_z = self.joints[1].c.inv_conditional_y(w, generated_y)
        
        marginalized_x = self.marginals[0].quantile(rng_x)
        marginalized_y = self.marginals[1].quantile(generated_y)
        marginalized_z = self.marginals[2].quantile(generated_z)
        
        return (marginalized_x, marginalized_y, marginalized_z)
