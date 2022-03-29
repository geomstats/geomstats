"""Benchmark metric inner product."""

from importlib import import_module

import pandas as pd
import pytest


def read_benchmark_inner_product_data():
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
        inner_product_kwargs,
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
        tangent_vec_a = manifold.random_tangent_vec(n_samples, base_point)
        tangent_vec_b = manifold.random_tangent_vec(n_samples, base_point)
        inner_product_args = (tangent_vec_a, tangent_vec_b, base_point)
        data.append((metric, inner_product_args, inner_product_kwargs))

    return data, ids


benchmark_data, benchmark_ids = read_benchmark_inner_product_data()


@pytest.mark.parametrize(
    "metric, inner_product_args, inner_product_kwargs",
    benchmark_data,
    ids=benchmark_ids,
)
def test_benchmark_inner_product(
    metric, inner_product_args, inner_product_kwargs, benchmark
):
    """Benchmark metric inner product.

    Parameters
    ----------
    metric : object
        Metric object.
    inner_product_args : tuple
        Arguments to log function.
    inner_product_kwargs : tuple
        Keyword arguments to log function.
    """
    benchmark.pedantic(
        metric.inner_product,
        args=inner_product_args,
        kwargs=inner_product_kwargs,
        iterations=10,
        rounds=10,
    )
