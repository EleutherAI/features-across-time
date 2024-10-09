from pathlib import Path

import torch
from torch import Tensor
from concept_erasure import QuadraticEraser, OracleEraser, groupby
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def test_vision_datasets():
    ds_path = Path.cwd() / "data" /f'low_mu_high_sigma.hf'

    ds = load_from_disk(str(ds_path))
    ds.set_format("torch", columns=["pixel_values", "label"])

    X, Y = ds['pixel_values'].flatten(1, 3), ds['label']
    # take a sample of the Xs because it's too high-dimensional to quadratic regress on.
    # coordinates are all sampled in the same way so this doesn't change the results
    X = X[:, :100]

    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), 
        LogisticRegression(max_iter=10_000, tol=1e-2, solver='saga')
    )

    # Learns something from data
    lr = model.fit(X.numpy(), Y.numpy())
    beta = torch.from_numpy(lr.named_steps['logisticregression'].coef_)
    assert beta.norm(p=torch.inf) > 0.1

    # Remove quadratic information then add back linear differences
    eraser = QuadraticEraser.fit(X, Y)
    X_ = eraser(X, Y)
    groups: list[Tensor] = groupby(X_, Y).groups
    X_ = torch.cat([
        groups[0] - groups[0].mean() + eraser.class_means[0],
        groups[1] - groups[1].mean() + eraser.class_means[1]
    ])

    # Learns barely anything from remaining linear information
    l_lr = model.fit(X_.numpy(), Y.numpy())
    l_beta = torch.from_numpy(l_lr.named_steps['logisticregression'].coef_)
    assert l_beta.norm(p=torch.inf) < 1e-4

    # Remove linear information using OLEACE
    eraser = OracleEraser.fit(X, Y)
    X_ = eraser(X, Y)

    # Learns something from remaining quadratic information
    q_lr = model.fit(X_.numpy(), Y.numpy())
    q_beta = torch.from_numpy(q_lr.named_steps['logisticregression'].coef_)
    assert q_beta.norm(p=torch.inf) > 0.1

    # Learns more from quadratic information than linear information
    assert q_beta.norm(p=torch.inf) > l_beta.norm(p=torch.inf)

    # Working notes
    # scipy function supports truncnorm, wrapper around fortran code
    # for 2L (& maybe 3L) mlps, if you qleace the data it can't learn anything. but for sufficintly wide mlps there's reason to think
    # it might be wrong.
    # Figure out the width at which this changes
    # Past experiments were on CIFAR10
    # Less naive: use MLP mixer: downsample with maxpooling, make it thicker using same MLP applied to every pixel, transpose
    # and then apply the MLP channelwise. 
    # Bilinear MLP is maybe a bit better than GeLU MLP, and it is quadratic so QLEACE provably makes it impossible to learn
    # it, and GELU is plausibly in a similar class of functions to bilinear.


if __name__ == "__main__":
    # Slow due to fitting quadratic models on large datasets
    test_vision_datasets()