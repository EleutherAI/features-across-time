import torch

class DuryDistribution:
    """Sample from p = exp(-a - b * x)
    
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
    def __init__(self, device: str, mu: torch.Tensor):
        """Sampler for a maximum entropy distribution subject to hypercube and mean constraints.
        mu: mean of distribution."""
        def get_mu(b: torch.Tensor) -> torch.Tensor:
            return -(b - b.exp() + 1)/(b * (b.exp() - 1))

        def newton(f, x0, tol=1.48e-8, max_iter=50):
            x = x0.clone().detach().requires_grad_(True)
            for _ in range(max_iter):
                y = f(x)
                if torch.max(torch.abs(y)) < tol:
                    break
                grad_y, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y, device=self.device), create_graph=True)
                x = x - y / grad_y
                x.detach_().requires_grad_()
            return x

        self.device = device
        self.b = newton(lambda b: get_mu(b) - mu.to(self.device), torch.full_like(mu, 0.1, device=self.device))
        self.a = -self.b + ((self.b.exp() - 1) / self.b).log()
    
    
    def sample(self, num_samples: int):
        u = torch.rand((num_samples), device=self.device)
        return (-1 / (self.b * u * self.a.exp() - 1)).log() / self.b