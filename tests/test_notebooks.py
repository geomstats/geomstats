"""Unit tests for the notebooks."""

import os
import glob
import subprocess
import tempfile
import json

import pytest


BACKEND = os.environ.get("GEOMSTATS_BACKEND", "numpy")


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


NOTEBOOKS_DIR = 'notebooks'
paths = glob.glob(f'{NOTEBOOKS_DIR}/*.ipynb')


@pytest.mark.parametrize('path', paths)
def test_notebook(path):
    with open(path, 'r') as file:
        metadata = json.load(file).get('metadata')

    all_backends = ['numpy', 'torch', 'tensorflow', 'autograd']
    backends = metadata.get('backends', all_backends)
    if BACKEND not in backends:
        pytest.skip()

    _exec_notebook(path)
