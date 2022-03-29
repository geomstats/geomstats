"""Benchmark metric logarithm."""

from importlib import import_module

import pandas as pd
import pytest


def read_benchmark_log_data():
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
        log_kwargs,
        manifold_args,
        metric_args,
    ) in params:
        ids.append(
            metric + " metric_args= " + str(metric_args) + " samples= " + str(n_samples)
        )

        module = import_module(module)
        manifold = getattr(module, manifold)(*manifold_args)
        metric = getattr(module, metric)(*metric_args)
        base_point = manifold.random_point(n_samples)
        point = manifold.random_point(n_samples)
        log_args = (point, base_point)
        data.append((metric, log_args, log_kwargs))

    return data, ids


benchmark_data, benchmark_ids = read_benchmark_log_data()


@pytest.mark.parametrize(
    "metric, log_args, log_kwargs", benchmark_data, ids=benchmark_ids
)
def test_benchmark_log(metric, log_args, log_kwargs, benchmark):
    """Benchmark metric logarithm map.

    Parameters
    ----------
    metric : object
        Metric object.
    log_args : tuple
        Arguments to log function.
    log_kwargs : tuple
        Keyword arguments to log function.
    """
    benchmark.pedantic(
        metric.log, args=log_args, kwargs=log_kwargs, iterations=10, rounds=10
    )
