import math

import torch


def kl_divergence_from_logits(
    p: torch.Tensor, q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Memory-efficient KL divergence between two sets of logits.
    Intermediate variables are (p.size(0), 1).

    Note that p.sub(logsumexp_p) is equivalent to converting p
    to log probabilities."""
    logsumexp_p = p.logsumexp(dim).unsqueeze(dim)
    logsumexp_q = q.logsumexp(dim).unsqueeze(dim)

    return torch.nansum(
        p.sub(logsumexp_p).exp() * (p.sub(logsumexp_p) - q.sub(logsumexp_q)),
        dim,
    )


def kl_divergence_from_probabilities(
    p: torch.Tensor, q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Compute the KL divergence between two sets of probabilities."""

    return torch.nansum(p * (p / q).log(), dim)


def js_divergence_from_logits(
    logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence between two sets of logits"""
    logsumexp_p = logit_p.logsumexp(dim).unsqueeze(dim)
    logsumexp_q = logit_q.logsumexp(dim).unsqueeze(dim)

    # Mean of P and Q
    log_m = (
        torch.stack([logit_p - logsumexp_p, logit_q - logsumexp_q])
        .sub(math.log(2))
        .logsumexp(0)
    )

    kl_p = torch.nansum(
        logit_p.sub(logsumexp_p).exp() * (logit_p.sub(logsumexp_p).sub(log_m)), dim
    )
    kl_q = torch.nansum(
        logit_q.sub(logsumexp_q).exp() * (logit_q.sub(logsumexp_q).sub(log_m)), dim
    )
    return 0.5 * (kl_p + kl_q)


def one_hot_js_divergence(
    logit_q: torch.Tensor, p_index: torch.Tensor, batch: int, dim: int = -1
) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence between
    a one-hot vector and a set of logits."""
    logsumexp_q = logit_q.logsumexp(-1, keepdim=True)

    # accumulate log_m (starts in linear space)
    log_m = logit_q.sub(logsumexp_q).sub(math.log(2)).exp()
    log_m[torch.arange(batch * 2048), p_index] += 0.5
    log_m += torch.finfo(torch.float32).eps
    log_m = log_m.log()

    # p * log(p / m) at p = 1 -> log(p) - log(m) = -log(m)
    kl_p = -log_m[torch.arange(batch * 2048), p_index]
    kl_q = torch.nansum(
        logit_q.sub(logsumexp_q).exp() * (logit_q.sub(logsumexp_q).sub(log_m)), dim
    )
    return 0.5 * (kl_p + kl_q)
