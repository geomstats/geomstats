"""Unit tests for the functions manifolds."""

import math
import warnings

import numpy as np

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.functions import L2Space, SinfSpace


class TestFunctions(geomstats.tests.TestCase):
    def setup_method(self):
        domain = gs.linspace(-math.pi, math.pi)
        self.f = gs.sin(domain)
        self.f_sinf = self.f / np.trapz(self.f, domain)
        self.L2 = L2Space(domain)
        self.Sinf = SinfSpace(domain)

    def test_belongs(self):
        result = self.L2.belongs(self.f)
        self.assertTrue(result)
        result = self.Sinf(self.f_sinf)
        self.assertTrue(result)
