import csv
from ast import literal_eval
from importlib import import_module

import pytest


def read_benchmark_exp_data():
    data = []
    ids = []
    with open("benchmark_exp.csv") as csvfile:
        for (
            manifold,
            metric,
            manifold_args,
            metric_args,
            exp_kwargs,
            module,
            n_samples,
        ) in csv.reader(csvfile):
            ids.append(
                metric + "metric_args= " + metric_args + "samples= " + str(n_samples)
            )
            module = import_module(module)
            manifold = getattr(module, manifold)(*literal_eval(manifold_args))
            metric = getattr(module, metric)(*literal_eval(metric_args))
            base_point = manifold.random_point(n_samples)
            tangent_vec = manifold.random_tangent_vec(n_samples)
            exp_args = (tangent_vec, base_point)
            data.append(metric, exp_args, exp_kwargs)

    return (data, ids)


data, ids = read_benchmark_exp_data()


@pytest.mark.parametrize("metric, metric_args, exp_args, exp_kwargs", data, ids=ids)
def test_benchmark_exp(metric, exp_args, exp_kwargs, benchmark):
    benchmark.pedantic(
        metric.exp, args=exp_args, kwargs=exp_kwargs, iterations=10, rounds=10
    )
