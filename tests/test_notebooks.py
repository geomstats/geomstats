"""Unit tests for the notebooks."""

import subprocess
import tempfile

import geomstats.tests


def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix='.ipynb') as fout:
        args = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                '--ExecutePreprocessor.timeout=1000',
                '--output', fout.name, path]
        subprocess.check_call(args)


class TestNotebooks(geomstats.tests.TestCase):
    @staticmethod
    @geomstats.tests.np_only
    def test_tangent_pca_s2():
        _exec_notebook('notebooks/tangent_pca_s2.ipynb')

    @staticmethod
    def test_01_data_on_manifolds():
        _exec_notebook('notebooks/01_data_on_manifolds.ipynb')
