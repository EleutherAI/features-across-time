from torch import nn, optim, Tensor
import torch


def koleo(x: Tensor) -> Tensor:
    """Kozachenko-Leonenko estimator of entropy."""
    return torch.cdist(x, x).kthvalue(2).values.log().mean()


def psd_sqrt(A: Tensor) -> Tensor:
    """Compute the unique p.s.d. square root of a positive semidefinite matrix."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)
    return U * L.sqrt() @ U.mH


def truncated_normal(
    n: int,
    mean: Tensor,
    cov: Tensor,
    *,
    koleo_weight: float = 1e-3,
    max_iter: int = 100,
    seed: int = 0,
):
    """Generate `n` samples from max-ent distribution on [0, 1]ˆd with given moments."""
    d = mean.shape[-1]
    eps = torch.finfo(mean.dtype).eps
    rng = torch.Generator(device=mean.device).manual_seed(seed)

    # Initialize with max-ent samples matching `mean` and `cov` but without hypercube
    # constraint. We do so in a way that is robust to singular `cov`
    z = mean.new_empty([n, d]).normal_(generator=rng)
    x = torch.clamp(z @ psd_sqrt(cov) + mean, eps, 1 - eps) # Project into hypercube

    # Reparametrize to enforce hypercube constraint
    z = nn.Parameter(x.logit())
    opt = optim.LBFGS([z], line_search_fn="strong_wolfe", max_iter=max_iter)
    
    def closure():
        opt.zero_grad()
        x = z.sigmoid()
        loss = torch.norm(x.mean(0) - mean) + torch.norm(x.T.cov() - cov)
        loss -= koleo_weight * koleo(x)
        loss.backward()
        return float(loss)

    opt.step(closure)
    return z.sigmoid().detach()