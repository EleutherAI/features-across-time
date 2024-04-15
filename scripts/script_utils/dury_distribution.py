from tqdm import tqdm
import time
import torch

class DuryDistribution:
    """Sample from exp(-a - b * x)
    
    # Derivations for helper equations
    a, x, mu, u = symbols('a x mu u')
    b = symbols('b', nonzero=True)
    p = exp(-a - b * x)

    def derive_mu() -> Callable:
        total = integrate(p, (x, 0, 1))
        a_expr = solve(total - 1, a)[0]
        mu = integrate(x * p.subs(a, a_expr), (x, 0, 1)).simplify()
        return lambdify(b, mu)

    def derive_quantile() -> Callable:
        cdf = integrate(p, (x, 0, x))
        inverse_cdf = solve(Eq(cdf, u), x)[0]
        return lambdify((a, b, u), inverse_cdf)
    """
    def __init__(self, device: str):
        self.device = device

    def quantile(self, a: torch.Tensor, b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (-1 / (b * u * a.exp() - 1)).log() / b

    def mu(self, b: torch.Tensor) -> torch.Tensor:
        return -(b - b.exp() + 1)/(b * (b.exp() - 1))
    
    def get_sampler(self, mu: torch.Tensor):
        """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
        Input: mu; mean of distribution.
        """
        b = self.newton(lambda b: self.mu(b) - mu, torch.full_like(mu, 0.1, device=self.device))
        a = -b + ((b.exp() - 1) / b).log()
        return lambda: self.quantile(a, b, torch.rand((1), device=self.device))

    def newton(self, f, x0, tol=1.48e-8, max_iter=50):
        x = x0.clone().detach().requires_grad_(True)
        for _ in range(max_iter):
            y = f(x)
            if torch.max(torch.abs(y)) < tol:
                break
            grad_y, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y, device=self.device), create_graph=True)
            x = x - y / grad_y
            x.detach_().requires_grad_()
        return x