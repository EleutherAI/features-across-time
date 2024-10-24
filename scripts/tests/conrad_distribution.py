import numpy as np

from scripts.script_utils.conrad_distribution import ConradDistribution


def test_conrad_distribution():
    mu = np.random.uniform(0, 1, (100, 100))
    dd = ConradDistribution(mu)
    samples = dd.sample(30_000)

    assert np.all(np.isfinite(samples))
    assert np.allclose(samples.mean(0), mu, atol=9e-3)
    assert np.all(samples <= 1.0) and np.all(samples >= 0.0)


if __name__ == "__main__":
    test_conrad_distribution()
