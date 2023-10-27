"""Unit tests for the notebooks."""

import glob

from geomstats.test.parametrizers import NotebooksParametrizer
from geomstats.test.test_case import TestCase


class NotebooksTestData:
    def __init__(self):
        NOTEBOOKS_DIR = "notebooks"
        self.paths = sorted(glob.glob(f"{NOTEBOOKS_DIR}/*.ipynb"))


class TestNotebooks(TestCase, metaclass=NotebooksParametrizer):
    testing_data = NotebooksTestData()
