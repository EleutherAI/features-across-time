from numpy.typing import NDArray
from scipy.optimize import newton
import numpy as np


class DuryDistribution:
    """Sample from p = exp(-a - b * x) in range [0, 1].
    
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
    def __init__(self, mu: NDArray):
        """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
        mu: mean of distribution."""
        eps = np.finfo(mu.dtype).eps

        def get_mu(b: NDArray) -> NDArray:
            num = -(b - np.expm1(b))
            denom = b * (np.expm1(b))
            return num / (denom + eps)
        
        clipped_mu = mu.clip(3e-3, 1 - 3e-3)
        self.b = newton(
            lambda b: get_mu(b) - clipped_mu, np.full_like(clipped_mu, 0.001), maxiter=20_000, tol=1e-5
        )

        mask = ~np.isfinite(self.b)
        while mask.sum():
            self.b[mask] = newton(
                lambda b: get_mu(b) - clipped_mu, np.full_like(clipped_mu, np.random.uniform(1e-4, 1e-2)), maxiter=20_000, tol=1e-3
            )[mask]
            mask = ~np.isfinite(self.b)
        self.a = -self.b + np.log(np.expm1(self.b) / self.b)
    
    
    def sample(self, num_samples: int):
        '''Generate num_samples samples'''
        u = np.random.rand(num_samples, *self.a.shape)
        samples = np.log(-1 / ((self.b * u) * np.exp(self.a) - 1)) / self.b[None, :]
        return samples