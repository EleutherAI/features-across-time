from numpy.typing import NDArray
from scipy.optimize import newton
import numpy as np

class DuryDistribution:
    """Sample from p = exp(-a - b * x) in range [0, 1]
    
    # Derivations for helper equations
    a, x, mu, u = symbols('a x mu u')
    b = symbols('b', nonzero=True)
    p = exp(-a - b * x)

    def derive_mu() -> Callable:
        total = integrate(p, (x, 0, 1))
        a_expr = solve(total - 1, a)[0]
        mu = integrate(x * p.subs(a, a_expr), (x, 0, 1)).simplify()
        return lambdify(b, mu)

    def derive_sample() -> Callable:
        cdf = integrate(p, (x, 0, x))
        inverse_cdf = solve(Eq(cdf, u), x)[0]
        return lambdify((a, b, u), inverse_cdf)
    """
    def __init__(self, mu: NDArray, start: int = 0.1):
        """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
        mu: mean of distribution."""
        def get_mu(b: NDArray) -> NDArray:
            num = -(b - np.expm1(b))
            denom = b * (np.expm1(b) + np.finfo(b.dtype).eps)
            return num / denom

        def get_mu_deriv(b: NDArray) -> NDArray:
            num = b ** 2 * np.expm1(b) + 1 - np.expm1(2*b) + 2*np.expm1(b) + 2
            denom = b ** 2 * (np.expm1(2*b) + 1 - 2*np.expm1(b) + 2)
            return num / denom
        
        self.b, converged, _ = newton(lambda b: get_mu(b) - mu, np.full_like(mu, start), fprime=get_mu_deriv, maxiter=200_000, full_output=True)
        self.a = -self.b + np.log(np.expm1(self.b) / self.b)
        
        print(f"Number of failures to converge: {converged.size - np.sum(converged)} / {converged.size}")
    
    
    def sample(self, num_samples: int):
        '''Generate num_samples samples for each mu'''
        u = np.random.rand(num_samples, *self.a.shape)
        samples = np.log(-1 / ((self.b * u) * np.exp(self.a) - 1)) / self.b[None, :]
        return samples


if __name__ == "__main__":
    mu = np.random.rand(3, 5, 6)
    dd = DuryDistribution(mu)
    samples = dd.sample(30)
    print(np.sum(np.isnan(samples)), "nan samples")