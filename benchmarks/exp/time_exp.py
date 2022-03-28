"""Benchmark metric exponential."""

from importlib import import_module

import pandas as pd
import pytest


def read_benchmark_exp_data():
    """Read benchmark parameters from file benchmark_params.pkl."""
    data = []
    ids = []
    df = pd.read_pickle("./benchmark_params.pkl")
    params = list(df.itertuples(index=False))
    for (
        manifold,
        module,
        metric,
        n_samples,
        exp_kwargs,
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
        tangent_vec = manifold.random_tangent_vec(n_samples, base_point)
        exp_args = (tangent_vec, base_point)
        data.append((metric, exp_args, exp_kwargs))

    return (data, ids)


benchmark_data, benchmark_ids = read_benchmark_exp_data()


@pytest.mark.parametrize(
    "metric, exp_args, exp_kwargs", benchmark_data, ids=benchmark_ids
)
def test_benchmark_exp(metric, exp_args, exp_kwargs, benchmark):
    """Benchmark metric exponential map.

    Parameters
    ----------
    metric : object
        Metric object.
    exp_args : tuple
        Arguments to exp function.
    exp_kwargs : tuple
        Keyword argumetns to exp function.
    """
    benchmark.pedantic(
        metric.exp, args=exp_args, kwargs=exp_kwargs, iterations=10, rounds=10
    )
