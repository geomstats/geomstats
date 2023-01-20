import glob
import importlib
import os
import warnings

import matplotlib
import pytest
from matplotlib import pyplot as plt

matplotlib.use("Agg")


BACKEND = os.environ.get("GEOMSTATS_BACKEND", "numpy")
ALL_BACKENDS = ["numpy", "autograd", "pytorch"]
AUTODIFF_BACKENDS = ALL_BACKENDS[1:]
NP_LIKE_BACKENDS = ALL_BACKENDS[:2]

EXAMPLES_DIR = "examples"

paths = sorted(glob.glob(f"{EXAMPLES_DIR}/*.py"))
SKIP = {"geomstats_in_pymanopt"}

METADATA = {
    "gradient_descent_s2": {
        "backends": NP_LIKE_BACKENDS,
        "kwargs": {"max_iter": 64, "output_file": None},
    },
    "geodesic_regression_hypersphere": AUTODIFF_BACKENDS,
    "geodesic_regression_se2": ["autograd"],
    "geodesic_regression_grassmannian": ["autograd"],
    "learning_graph_embedding_and_predicting": ALL_BACKENDS[:-1],
}
np_like_backends = [
    "empirical_frechet_mean_uncertainty_sn",
    "learning_graph_structured_data_h2",
    "plot_bch_so3",
    "plot_agglomerative_hierarchical_clustering_s2",
    "plot_expectation_maximization_ball",
    "plot_kernel_density_estimation_classifier_s2",
    "plot_kmeans_manifolds",
    "plot_kmedoids_manifolds",
    "plot_knn_s2",
    "plot_online_kmeans_s1",
    "plot_online_kmeans_s2",
    "tangent_pca_h2",
    "tangent_pca_s2",
    "tangent_pca_so3",
]
for example_name in np_like_backends:
    METADATA[example_name] = NP_LIKE_BACKENDS


@pytest.mark.parametrize("path", paths)
def test_example(path):
    warnings.simplefilter("ignore", category=UserWarning)

    example_name = path.split(os.sep)[-1].split(".")[0]
    metadata = METADATA.get(example_name, {})
    if not isinstance(metadata, dict):
        metadata = {"backends": metadata}

    backends = metadata.get("backends", ALL_BACKENDS)
    if example_name in SKIP or BACKEND not in backends:
        pytest.skip()

    spec = importlib.util.spec_from_file_location("module.name", path)

    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    kwargs = metadata.get("kwargs", {})
    example.main(**kwargs)
    plt.close("all")
