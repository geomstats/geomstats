"""Benchmarking parameters generation file."""

import argparse
from itertools import chain, product

import pandas as pd

parser = argparse.ArgumentParser(
    description="Generate parameters for which benchmark is run"
)

parser.add_argument(
    "-m",
    "--manifold",
    type=str,
    default="all",
    help="Manifold for which benchmark is run. 'all' denotes all manifolds present.",
)
parser.add_argument(
    "-n",
    "--n_samples",
    type=int,
    default=10,
    help="Number of samples for which benchmark is run",
)
args = parser.parse_args()


def spd_manifold_params(n_samples):
    """Generate spd manifold benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "SPDMatrices"
    manifold_args = [(2,), (5,), (10,)]
    module = "geomstats.geometry.spd_matrices"

    def spd_affine_metric_params():
        params = []
        metric = "SPDMetricAffine"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product([item for item, in manifold_args], power_args))
        manifold_args_re = [
            item for item in manifold_args for i in range(len(power_args))
        ]
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args_re, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    def spd_bures_wasserstein_metric_params():
        params = []
        metric = "SPDMetricBuresWasserstein"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    def spd_euclidean_metric_params():
        params = []
        metric = "SPDMetricEuclidean"
        power_args = [-0.5, 1, 0.5]
        metric_args = list(product([item for item, in manifold_args], power_args))
        manifold_args_re = [
            item for item in manifold_args for i in range(len(power_args))
        ]
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args_re, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    def spd_log_euclidean_metric_params():
        params = []
        metric = "SPDMetricLogEuclidean"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return list(
        chain(
            *[
                spd_bures_wasserstein_metric_params(),
                spd_affine_metric_params(),
                spd_euclidean_metric_params(),
                spd_log_euclidean_metric_params(),
            ]
        )
    )


def stiefel_params(n_samples):
    """Generate stiefel benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Stiefel"
    manifold_args = [(3, 2), (4, 3)]
    module = "geomstats.geometry.stiefel"

    def stiefel_canonical_metric_params():
        params = []
        metric = "StiefelCanonicalMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return stiefel_canonical_metric_params()


def pre_shape_params(n_samples):
    """Generate pre shape benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "PreShapeSpace"
    manifold_args = [(3, 3), (5, 5)]
    module = "geomstats.geometry.pre_shape"

    def pre_shape_metric_params():
        params = []
        metric = "PreShapeMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return pre_shape_metric_params()


def positive_lower_triangular_matrices_params(n_samples):
    """Generate positive lower triangular matrices benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "PositiveLowerTriangularMatrices"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.positive_lower_triangular_matrices"

    def cholesky_metric_params():
        params = []
        metric = "CholeskyMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return cholesky_metric_params()


def minkowski_params(n_samples):
    """Generate minkowski benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Minkowski"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.minkowski"

    def minkowski_metric_params():
        params = []
        metric = "MinkowskiMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return minkowski_metric_params()


def matrices_params(n_samples):
    """Generate matrices benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Matrices"
    manifold_args = [(3, 3), (5, 5)]
    module = "geomstats.geometry.matrices"

    def matrices_metric_params():
        params = []
        metric = "MatricesMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return matrices_metric_params()


def hypersphere_params(n_samples):
    """Generate hypersphere benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Hypersphere"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.hypersphere"

    def hypersphere_metric_params():
        params = []
        metric = "HypersphereMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return hypersphere_metric_params()


def grassmanian_params(n_samples):
    """Generate grassmanian parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Grassmannian"
    manifold_args = [(4, 3), (5, 4)]
    module = "geomstats.geometry.grassmannian"

    def grassmannian_canonical_metric_params():
        params = []
        metric = "GrassmannianCanonicalMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return grassmannian_canonical_metric_params()


def full_rank_correlation_matrices_params(n_samples):
    """Generate full rank correlation matrices benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "FullRankCorrelationMatrices"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.full_rank_correlation_matrices"

    def full_rank_correlation_affine_quotient_metric_params():
        params = []
        metric = "FullRankCorrelationAffineQuotientMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return full_rank_correlation_affine_quotient_metric_params()


def hyperboloid_params(n_samples):
    """Generate hyperboloid benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "Hyperboloid"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.hyperboloid"

    def hyperboloid_metric_params():
        params = []
        metric = "HyperboloidMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return hyperboloid_metric_params()


def poincare_ball_params(n_samples):
    """Generate poincare ball benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "PoincareBall"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.poincare_ball"

    def poincare_ball_metric_params():
        params = []
        metric = "PoincareBallMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return poincare_ball_metric_params()


def poincare_half_space_params(n_samples):
    """Generate poincare half space benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "PoincareHalfSpace"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.poincare_half_space"

    def poincare_half_space_metric_params():
        params = []
        metric = "PoincareHalfSpaceMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return poincare_half_space_metric_params()


def poincare_polydisk_params(n_samples):
    """Generate poincare polydisk benchmarking parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to be used.

    Returns
    -------
    _ : list.
        List of params.
    """
    manifold = "PoincarePolydisk"
    manifold_args = [(3,), (5,)]
    module = "geomstats.geometry.poincare_polydisk"

    def poincare_poly_disk_metric_params():
        params = []
        metric = "PoincarePolydiskMetric"
        metric_args = manifold_args
        kwargs = {}
        common = manifold, module, metric, n_samples, kwargs
        for manifold_arg, metric_arg in zip(manifold_args, metric_args):
            params += [common + (manifold_arg, metric_arg)]
        return params

    return poincare_poly_disk_metric_params()


manifolds = [
    "spd_manifold",
    "stiefel",
    "pre_shape",
    "positive_lower_triangular_matrices",
    "minkowski",
    "matrices",
    "hypersphere",
    "grassmanian",
    "hyperboloid",
    "poincare_ball",
    "poincare_half_space",
]


def generate_benchmark_params(manifold="all", n_samples=10):
    """Generate parameters for benchmarking.

    Parameters
    ----------
    manifold : str
        Manifold name or all.
        Optional, default "all".
    n_samples : int
        Number of samples.
        Optional, default 10.
    """
    params_list = []
    manifolds_list = manifolds if manifold == "all" else [manifold]
    params_list = [
        globals()[manifold + "_params"](n_samples) for manifold in manifolds_list
    ]
    params_list = list(chain(*params_list))
    df = pd.DataFrame(
        params_list,
        columns=[
            "manifold",
            "module",
            "metric",
            "n_samples",
            "exp_kwargs",
            "manifold_args",
            "metric_args",
        ],
    )
    df.to_pickle("benchmark_params.pkl")
    print("Generated params at benchmark_params.pkl.")


def main():
    """Generate Benchmark Params."""
    generate_benchmark_params(args.manifold, args.n_samples)


if __name__ == "__main__":
    main()
