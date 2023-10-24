import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import HypersphereIntrinsicRandomDataGenerator
from geomstats.test_cases.geometry.hypersphere import (
    HypersphereCoordsTransformTestCase,
    HypersphereExtrinsicTestCase,
    HypersphereIntrinsicTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.hypersphere import (
    Hypersphere2ExtrinsicMetricTestData,
    Hypersphere2IntrinsicMetricTestData,
    Hypersphere4ExtrinsicMetricTestData,
    HypersphereCoordsTransformTestData,
    HypersphereExtrinsicMetricTestData,
    HypersphereExtrinsicTestData,
    HypersphereIntrinsicTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        3,
    ],
)
def coords_transform_spaces(request):
    dim = request.param
    request.cls.space_intrinsic = Hypersphere(dim, default_coords_type="intrinsic")
    request.cls.space = request.cls.space_extrinsic = Hypersphere(
        dim, default_coords_type="extrinsic"
    )


@pytest.mark.usefixtures("coords_transform_spaces")
class TestHypersphereCoordsTransform(
    HypersphereCoordsTransformTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereCoordsTransformTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def extrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, default_coords_type="extrinsic", equip=True)


@pytest.mark.usefixtures("extrinsic_spaces")
class TestHypersphereExtrinsic(
    HypersphereExtrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereExtrinsicTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
    ],
)
def intrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, default_coords_type="intrinsic")


@pytest.mark.usefixtures("intrinsic_spaces")
class TestHypersphereIntrinsic(
    HypersphereIntrinsicTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereIntrinsicTestData()


@pytest.fixture(
    scope="class",
    params=[
        1,
        2,
        random.randint(3, 5),
    ],
)
def equipped_extrinsic_spaces(request):
    dim = request.param
    request.cls.space = Hypersphere(dim, default_coords_type="extrinsic")


@pytest.mark.usefixtures("equipped_extrinsic_spaces")
class TestHypersphereExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = HypersphereExtrinsicMetricTestData()

    @pytest.mark.random
    def test_sectional_curvature_is_one(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.ones(batch_shape)

        self.assertAllClose(res, expected, atol=atol)


class TestHypersphere2IntrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(2, default_coords_type="intrinsic", equip=False)
    space.equip_with_metric(HypersphereMetric)

    data_generator = HypersphereIntrinsicRandomDataGenerator(space)
    testing_data = Hypersphere2IntrinsicMetricTestData()

    def test_riemann_tensor_spherical_coords(self, base_point):
        """Test the Riemann tensor on the sphere.

        riemann_tensor[...,i,j,k,l] = R_{ijk}^l
            Riemannian tensor curvature,
            with the contravariant index on the last dimension.

        Note that the base_point is input in spherical coordinates.

        Expected formulas taken from:
        https://digitalcommons.latech.edu/cgi/viewcontent.cgi?
        article=1008&context=mathematics-senior-capstone-papers
        """
        riemann_tensor_ijk_l = self.space.metric.riemann_tensor(base_point)

        theta, _ = base_point[0], base_point[1]
        expected_212_1 = gs.array(gs.sin(theta) ** 2)
        expected_221_1 = gs.array(-gs.sin(theta) ** 2)
        expected_121_2 = gs.array(1.0)
        expected_112_2 = gs.array(-1.0)
        result_212_1 = riemann_tensor_ijk_l[1, 0, 1, 0]
        result_221_1 = riemann_tensor_ijk_l[1, 1, 0, 0]
        result_121_2 = riemann_tensor_ijk_l[0, 1, 0, 1]
        result_112_2 = riemann_tensor_ijk_l[0, 0, 1, 1]
        self.assertAllClose(expected_212_1, result_212_1)
        self.assertAllClose(expected_221_1, result_221_1)
        self.assertAllClose(expected_121_2, result_121_2)
        self.assertAllClose(expected_112_2, result_112_2)


@pytest.mark.smoke
class TestHypersphere2ExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(2, default_coords_type="extrinsic", equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere2ExtrinsicMetricTestData()


@pytest.mark.smoke
class TestHypersphere4ExtrinsicMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = space = Hypersphere(4, default_coords_type="extrinsic", equip=False)
    space.equip_with_metric(HypersphereMetric)

    testing_data = Hypersphere4ExtrinsicMetricTestData()

    def test_exp_and_dist_and_projection_to_tangent_space(self, vector, base_point):
        tangent_vec = self.space.to_tangent(vector=vector, base_point=base_point)
        exp = self.space.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.space.metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)
