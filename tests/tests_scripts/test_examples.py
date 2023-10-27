import glob

import matplotlib

from geomstats.test.parametrizers import ExamplesParametrizer
from geomstats.test.test_case import TestCase

ALL_BACKENDS = ["numpy", "autograd", "pytorch"]
AUTODIFF_BACKENDS = ALL_BACKENDS[1:]
NP_LIKE_BACKENDS = ALL_BACKENDS[:2]

EXAMPLES_DIR = "examples"


matplotlib.use("Agg")


class ExamplesTestData:
    def __init__(self):
        self.paths = sorted(glob.glob(f"{EXAMPLES_DIR}/*.py"))

        self.skips = {"geomstats_in_pymanopt"}

        self.metadata = {
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
            self.metadata[example_name] = NP_LIKE_BACKENDS


class TestExamples(TestCase, metaclass=ExamplesParametrizer):
    testing_data = ExamplesTestData()
