import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.numerics.path import UniformlySampledPathEnergy
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase

from .data.path import UniformlySampledPathEnergyTestData


class TestUniformlySampledPathEnergy(TestCase, metaclass=DataBasedParametrizer):
    _dim = random.randint(2, 3)
    space = PoincareBall(_dim)
    path_energy = UniformlySampledPathEnergy(space)

    data_generator = RandomDataGenerator(space)
    testing_data = UniformlySampledPathEnergyTestData()

    def test_energy(self, path, expected, atol):
        res = self.path_energy.energy(path)
        self.assertAllClose(res, expected, atol=atol)

    def test_energy_per_time(self, path, expected, atol):
        res = self.path_energy.energy_per_time(path)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_dist_from_path_energy_per_time(self, n_points, n_times, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        geod_func = self.space.metric.geodesic(base_point, end_point=point)
        times = gs.linspace(0, 1.0, num=n_times)
        geod_points = geod_func(times)

        energy_per_time = self.path_energy.energy_per_time(geod_points)

        delta = 1 / (n_times - 1)
        dist_from_energy_per_time = gs.sum(
            gs.sqrt(energy_per_time * 2 / delta) * delta,
            axis=-1,
        )

        dist = self.space.metric.dist(point, base_point)
        self.assertAllClose(dist_from_energy_per_time, dist, atol=atol)
