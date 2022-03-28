from importlib import import_module

import pandas as pd
import pytest


def read_benchmark_exp_data():
    data = []
    ids = []
    df = pd.read_pickle("benchmark_exp.pkl")
    params = list(df.itertuples(index=False))
    print(params)
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
        point = manifold.random_point(n_samples)
        exp_args = (point, base_point)
        data.append((metric, exp_args, exp_kwargs))

    return (data, ids)


data, ids = read_benchmark_exp_data()


@pytest.mark.parametrize("metric, log_args, log_kwargs", data, ids=ids)
def test_benchmark_exp(metric, exp_args, exp_kwargs, benchmark):
    benchmark.pedantic(
        metric.log, args=exp_args, kwargs=exp_kwargs, iterations=10, rounds=10
    )
