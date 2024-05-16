import numpy as np

from scripts.script_utils.dury_distribution import DuryDistribution


def test_dury_distribution():
    mu = np.random.uniform(0, 1, (100, 100))
    dd = DuryDistribution(mu)
    samples = dd.sample(30_000)

    assert np.all(np.isfinite(samples))
    assert np.allclose(samples.mean(0), mu, atol=9e-3)


if __name__ == "__main__":
    test_dury_distribution()
    