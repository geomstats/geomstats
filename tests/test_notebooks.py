"""Unit tests for the notebooks."""

import subprocess
import tempfile

import geomstats.tests


def _exec_notebook(path):

    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


class TestNotebooks(geomstats.tests.TestCase):
    @staticmethod
    def test_01_data_on_manifolds():
        _exec_notebook("notebooks/01_data_on_manifolds.ipynb")

    @staticmethod
    def test_02_from_vector_spaces_to_manifolds():
        _exec_notebook("notebooks/02_from_vector_spaces_to_manifolds.ipynb")

    @staticmethod
    @geomstats.tests.np_autograd_and_torch_only
    def test_03_simple_machine_learning_on_tangent_spaces():
        _exec_notebook("notebooks/03_simple_machine_learning_on_tangent_spaces.ipynb")

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_04_riemannian_frechet_mean_and_tangent_pca():
        _exec_notebook("notebooks/04_riemannian_frechet_mean_and_tangent_pca.ipynb")

    @staticmethod
    @geomstats.tests.np_autograd_and_torch_only
    def test_05_riemannian_kmeans():
        _exec_notebook("notebooks/05_riemannian_kmeans.ipynb")

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_06_information_geometry():
        _exec_notebook("notebooks/06_information_geometry.ipynb")

    @staticmethod
    @geomstats.tests.autograd_and_torch_only
    def test_07_implement_your_own_riemannian_geometry():
        _exec_notebook("notebooks/07_implement_your_own_riemannian_geometry.ipynb")

    @staticmethod
    @geomstats.tests.np_only
    def test_usecase_visualizations_in_kendall_shape_spaces():
        _exec_notebook("notebooks/usecase_visualizations_in_kendall_shape_spaces.ipynb")

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_usecase_cell_shapes_analysis():
        _exec_notebook("notebooks/usecase_cell_shapes_analysis.ipynb")

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_usecase_emg_sign_classification_in_spd_manifold():
        _exec_notebook(
            "notebooks/usecase_emg_sign_classification_in_spd_manifold.ipynb"
        )

    @staticmethod
    @geomstats.tests.np_autograd_and_torch_only
    def test_usecase_graph_embedding_and_clustering_in_hyperbolic_space():
        _exec_notebook(
            "notebooks/"
            "usecase_graph_embedding_and_clustering_in_hyperbolic_space.ipynb"
        )

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_usecase_hand_poses_analysis_in_kendall_shape_space():
        _exec_notebook(
            "notebooks/" "usecase_hand_poses_analysis_in_kendall_shape_space.ipynb"
        )

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_usecase_optic_nerve_heads_analysis_in_kendall_shape_space():
        _exec_notebook(
            "notebooks/"
            "usecase_optic_nerve_heads_analysis_in_kendall_shape_space.ipynb"
        )

    @staticmethod
    @geomstats.tests.np_and_autograd_only
    def test_usecase_a03_connection_class():
        _exec_notebook("notebooks/" "usecase_a03_connection_class.ipynb")
