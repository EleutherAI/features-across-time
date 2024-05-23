import torch

from scripts.script_utils.truncated_normal import truncated_normal


def test_truncated_normal():
    mu = torch.rand((100,)) * 2
    cov = torch.eye(100) * 0.01
    samples = truncated_normal(20_000, mu, cov)

    assert torch.all(samples >= 0) and torch.all(samples <= 1)

    torch.testing.assert_close(samples.mean(dim=0), mu.clamp(0, 1), atol=0.1, rtol=10)
    torch.testing.assert_close(samples.T.cov(), cov, atol=0.1, rtol=10)


if __name__ == "__main__":
    test_truncated_normal()
    