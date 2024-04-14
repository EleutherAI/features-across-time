from scipy.optimize import newton
from typing import Callable
from sympy import symbols, exp, solve, integrate, lambdify, init_printing, oo, Eq, Integral
import math
from numpy.random import uniform

init_printing(use_latex=True)

class DuryDistribution:
    """p = exp(-a - b * x)"""
    get_mean: Callable[[float], float]
    get_quantile: Callable[[float], float]

    def __init__(self):
        a, x, mu = symbols('a x mu')
        b = symbols('b', nonzero=True)
        p = exp(-a - b * x)

        def derive_mean_function() -> Callable:
            total = integrate(p, (x, 0, 1))
            a_expr = solve(total - 1, a)[0]
            mu = integrate(x * p.subs(a, a_expr), (x, 0, 1)).simplify()
            return lambdify(b, mu, modules="numpy")

        def derive_quantile_function() -> Callable:
            cdf = integrate(p, (x, 0, x))
            inverse_cdf = solve(Eq(cdf, u), x)[0]
            return lambdify((a, b, u), inverse_cdf, modules='numpy')

        self.get_mean = derive_mean_function()
        self.get_quantile = derive_quantile_function()


    def get_sampler(self, mu: float):
        """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
        Input: mu; mean of distribution.
        """
        b = newton(lambda b: self.get_mean(b) - mu, 0.1)
        a = -b + math.log((exp(b) - 1) / b)
        return lambda: self.get_quantile(a, b, uniform(0, 1))


dd = DuryDistribution()
sampler = dd.get_sampler(0.5001)
sampler()

# class DuryDistributionTorch:
#     """p = exp(-a - b * x)"""
#     get_mean: Callable[[float], float]
#     get_quantile: Callable[[float], float]

#     def __init__(self):
#         a, x, mu = symbols('a x mu')
#         b = symbols('b', nonzero=True)
#         p = exp(-a - b * x)

#         def derive_mean_function() -> Callable:
#             total = integrate(p, (x, 0, 1))
#             a_expr = solve(total - 1, a)[0]
#             mu = integrate(x * p.subs(a, a_expr), (x, 0, 1)).simplify()
#             return lambdify(b, mu, modules="numpy")

#         def derive_quantile_function() -> Callable:
#             cdf = integrate(p, (x, 0, x))
#             inverse_cdf = solve(Eq(cdf, u), x)[0]
#             return lambdify((a, b, u), inverse_cdf, modules='numpy')

#         self.get_mean = derive_mean_function()
#         self.get_quantile = derive_quantile_function()


#     def get_sampler(self, mu: float):
#         """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
#         Input: mu; mean of distribution.
#         """
#         b = newton(lambda b: self.get_mean(b) - mu, 0.1)
#         a = -b + math.log((exp(b) - 1) / b)
#         return lambda: self.get_quantile(a, b, uniform(0, 1))
    