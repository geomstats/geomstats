"""Unit tests for the notebooks."""

import subprocess
import tempfile

import geomstats.tests


def _exec_notebook(path):

    file_name = tempfile.NamedTemporaryFile(suffix='.ipynb').name
    args = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--ExecutePreprocessor.timeout=1000',
            '--ExecutePreprocessor.kernel_name=python3',
            '--output', file_name, path]
    subprocess.check_call(args)


class TestNotebooks(geomstats.tests.TestCase):
    @staticmethod
    def test_01_data_on_manifolds():
        _exec_notebook('notebooks/01_data_on_manifolds.ipynb')

    @staticmethod
    def test_02_from_vector_spaces_to_manifolds():
        _exec_notebook('notebooks/02_from_vector_spaces_to_manifolds.ipynb')

    @staticmethod
    @geomstats.tests.np_and_pytorch_only
    def test_03_simple_machine_learning_tangent_spaces():
        _exec_notebook(
            'notebooks/03_simple_machine_learning_tangent_spaces.ipynb')

    @staticmethod
    @geomstats.tests.np_only
    def test_04_frechet_mean_and_tangent_pca():
        _exec_notebook('notebooks/04_frechet_mean_and_tangent_pca.ipynb')

    @staticmethod
    @geomstats.tests.np_and_pytorch_only
    def test_05_embedding_graph_structured_data_h2():
        _exec_notebook('notebooks/05_embedding_graph_structured_data_h2.ipynb')
