"""Benchmark metric distance."""

from importlib import import_module

import pandas as pd
import pytest


def read_benchmark_dist_data():
    """Read benchmark parameters from file benchmark_params.pkl."""
    data = []
    ids = []
    df = pd.read_pickle("benchmark_params.pkl")
    params = list(df.itertuples(index=False))
    for (
        manifold,
        module,
        metric,
        n_samples,
        dist_kwargs,
        manifold_args,
        metric_args,
    ) in params:
        ids.append(
            metric + " metric_args= " + str(metric_args) + " samples= " + str(n_samples)
        )

        module = import_module(module)
        manifold = getattr(module, manifold)(*manifold_args)
        metric = getattr(module, metric)(*metric_args)
        point_a = manifold.random_point(n_samples)
        point_b = manifold.random_point(n_samples)
        dist_args = (point_a, point_b)
        data.append((metric, dist_args, dist_kwargs))

    return data, ids


benchmark_data, benchmark_ids = read_benchmark_dist_data()


@pytest.mark.parametrize(
    "metric, dist_args, dist_kwargs", benchmark_data, ids=benchmark_ids
)
def test_benchmark_dist(metric, dist_args, dist_kwargs, benchmark):
    """Benchmark metric distance map.

    Parameters
    ----------
    metric : object
        Metric object.
    dist_args : tuple
        Arguments to dist function.
    dist_kwargs : tuple
        Keyword arguments to dist function.
    """
    benchmark.pedantic(
        metric.dist, args=dist_args, kwargs=dist_kwargs, iterations=10, rounds=10
    )
