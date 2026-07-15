CopulaGen

To use:
1. Define a distribution (Discrete,Continuous,ContinuousJoint,DVine)
2. Pass distribution into DataGenerator
3. Run DataGenerator.generate_points()
4. Profit

ContinuousJoint is strictly a 2d Continuous Joint Distribution - and is defined by two Continuous marginal distributions and a copula

For higher dimension data, use DVines - which are defined by a number of ContinuousJoint Distributions

To assign classes to data, use DataGenerator.classify()