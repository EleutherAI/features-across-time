from scipy.optimize import newton
from typing import Callable
from sympy import symbols, exp, solve, integrate, lambdify, init_printing, oo, Eq, Integral
import math
from numpy.random import uniform

init_printing(use_latex=True)

class DuryDistribution:
    get_mean: Callable[[float], float]
    get_inverse_cdf: Callable[[float], float]


    def __init__(self):
        def derive_mean_eq():
            """Documentation for the mean_fn used to sample 
            from the distribution"""

            a, x, mu = symbols('a x mu')
            b = symbols('b', nonzero=True)
            p = exp(-a - b * x)

            # get eq for a : int(p) = 1 in terms of b
            total = integrate(p, (x, 0, 1))
            a_expr = solve(total - 1, a)[0]

            # get eq for p_mean in [0, 1] in terms of b
            mu = integrate(x * p.subs(a, a_expr), (x, 0, 1)).simplify()
            mu = mu.simplify()
            
            # result:
            # -(b - exp(b) + 1)/(b*(exp(b) - 1)) 
            return lambdify(b, mu, modules="numpy")


        def derive_inverse_cdf():
            x, a, b, u = symbols('x a b u', real=True, positive=True)
            p = exp(-a - b * x) 

            cdf = integrate(p, (x, 0, x))
            inverse_cdf = solve(Eq(cdf, u), x)[0]
            
            return lambdify((a, b, u), inverse_cdf, modules='numpy')


        self.get_mean = derive_mean_eq()
        self.get_inverse_cdf = derive_inverse_cdf()


    def get_sampler(self, mu: float):
        """Sample from a maximum entropy distribution subject to 
        hypercube and mean constraints.
        Input: mu; mean of distribution.
        """
        b = newton(lambda b: self.get_mean(b) - mu, 0.1)
        a = -b + math.log((exp(b) - 1) / b)

        def cdf_sampler():
            u = uniform(0, 1)
            return self.get_inverse_cdf(a, b, u)
        
        return cdf_sampler
    

# dist = DuryDistribution()
# mu = 0.50001
# sampler = dist.get_sampler(mu)
# print(sampler())
# print(sampler())
