import math

import pytest

import geomstats.backend as gs

from .base import VectorSpaceOpenSetTestData
from .diffeo import DiffeoTestData
from .riemannian_metric import RiemannianMetricTestData


class PositiveRealsTestData(VectorSpaceOpenSetTestData):
    @pytest.mark.smoke
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[10.0]]), expected=[True]),
            dict(point=gs.array([[10 + 0j]]), expected=[True]),
            dict(point=gs.array([[10 + 1j]]), expected=[False]),
            dict(point=gs.array([[-10.0]]), expected=[False]),
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=[False, False, False],
            ),
            dict(point=gs.array([[1.0], [-1.5]]), expected=[True, False]),
        ]
        return data

    @pytest.mark.smoke
    def projection_test_data(self):
        data = [
            dict(point=gs.array([[1.0]]), expected=gs.array([[1.0]])),
            dict(
                point=gs.array([[-1.0]]),
                expected=gs.array([[gs.atol]]),
            ),
        ]
        return data


class PositiveRealsMetricTestData(RiemannianMetricTestData):
    @pytest.mark.smoke
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([1.0]),
                tangent_vec_b=gs.array([2.0]),
                base_point=gs.array([3.0]),
                expected=gs.array(2 / 9),
            ),
            dict(
                tangent_vec_a=gs.array([-2.0]),
                tangent_vec_b=gs.array([3.0]),
                base_point=gs.array([4.0]),
                expected=gs.array(-3 / 8),
            ),
        ]
        return data

    @pytest.mark.smoke
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[1.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[math.exp(1.0)]]),
            ),
            dict(
                tangent_vec=gs.array([[4.0]]),
                base_point=gs.array([[2.0]]),
                expected=gs.array([[2 * math.exp(2.0)]]),
            ),
            dict(
                tangent_vec=gs.array([[1.0], [2.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[math.exp(1.0)], [math.exp(2.0)]]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0]]),
                base_point=gs.array([[2.0]]),
                expected=gs.array([[2 * math.log(2.0)]]),
            ),
            dict(
                point=gs.array([[2.0]]),
                base_point=gs.array([[4.0]]),
                expected=gs.array([[-4 * math.log(2.0)]]),
            ),
            dict(
                point=gs.array([[1.0], [2.0]]),
                base_point=gs.array([[1.0]]),
                expected=gs.array([[0], [math.log(2.0)]]),
            ),
        ]
        return data


class LogDiffeoTestData(DiffeoTestData):
    @pytest.mark.smoke
    def diffeomorphism_test_data(self):
        data = [
            dict(base_point=gs.array([1.0]), expected=gs.array([0.0])),
            dict(
                base_point=gs.array([[2.0], [math.exp(1.0)]]),
                expected=gs.array([[math.log(2.0)], [1.0]]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def inverse_test_data(self):
        data = [
            dict(image_point=gs.array([0.0]), expected=gs.array([1.0])),
            dict(
                image_point=gs.array([[math.log(2.0)], [2.0]]),
                expected=gs.array([[2.0], [math.exp(2.0)]]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([3.0]),
                base_point=gs.array([2.0]),
                expected=gs.array([1.5]),
            ),
            dict(
                tangent_vec=gs.array([3.0]),
                image_point=gs.array([math.log(2.0)]),
                expected=gs.array([1.5]),
            ),
        ]
        return data

    @pytest.mark.smoke
    def inverse_tangent_test_data(self):
        data = [
            dict(
                image_tangent_vec=gs.array([1.5]),
                image_point=gs.array([math.log(2.0)]),
                expected=gs.array([3.0]),
            ),
            dict(
                image_tangent_vec=gs.array([1.5]),
                base_point=gs.array([2.0]),
                expected=gs.array([3.0]),
            ),
        ]
        return data
