"""Unit tests for the notebooks."""

import glob
import json
import os
import subprocess
import tempfile

import pytest

BACKEND = os.environ.get("GEOMSTATS_BACKEND", "numpy")
ALL_BACKENDS = ["numpy", "pytorch", "autograd"]


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


NOTEBOOKS_DIR = "notebooks"
paths = sorted(glob.glob(f"{NOTEBOOKS_DIR}/*.ipynb"))


@pytest.mark.parametrize("path", paths)
def test_notebook(path):
    with open(path, "r", encoding="utf8") as file:
        metadata = json.load(file).get("metadata")

    backends = metadata.get("backends", ALL_BACKENDS)
    if BACKEND not in backends:
        pytest.skip()

    _exec_notebook(path)
