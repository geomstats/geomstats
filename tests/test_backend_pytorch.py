"""
Unit tests for numpy backend.
"""

import os
import unittest
import warnings

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup


@geomstats.tests.pytorch_only
class TestBackendPytorch(geomstats.tests.TestCase):
    def test_sampling_choice(self):
        res = gs.random.choice(10, (5, 1, 3))
        self.assertAllClose(res.shape, [5, 1, 3])
