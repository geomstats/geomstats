"""Unit tests for the graphspace quotient space."""

import networkx as nx

import geomstats.backend as gs
from tests.conftest import Parametrizer, TestCase, np_backend
from tests.data.graph_space_data import (
    AlignerTestData,
    DecoratorsTestData,
    GraphSpaceMetricTestData,
    GraphSpaceTestData,
    GraphTestData,
    PointToGeodesicAlignerTestData,
)
from tests.stratified_test_cases import (
    PointSetMetricTestCase,
    PointSetTestCase,
    PointTestCase,
)

IS_NOT_NP = not np_backend()


def _expand_point(point):
    return gs.expand_dims(point, 0)


def _repeat_point(point, n_reps=2):
    return gs.repeat(_expand_point(point), n_reps, axis=0)


def _expand_and_repeat_point(point, n_reps=2):
    return _expand_point(point), _repeat_point(point, n_reps=n_reps)


class TestGraphSpace(PointSetTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
    testing_data = GraphSpaceTestData()

    def test_random_point_output_shape(self, space, n_samples):
        pts = space.random_point(n_samples)
        if n_samples == 1:
            self.assertTrue(pts.ndim == 2)
        else:
            self.assertTrue(pts.ndim == 3)
            self.assertTrue(pts.shape[0] == n_samples)

    def test_set_to_array_output_shape(self, space, points):
        self.assertTrue(space.set_to_array(points).shape == points.shape)

    def test_permute(self, space, graph, permutation, expected):
        permuted_graph = space.permute(graph, permutation)
        self.assertAllClose(permuted_graph, expected)

    def test_set_to_networkx(self, space, points):
        nx_objects = space.set_to_networkx(points)

        if type(nx_objects) is list:
            self.assertTrue(nx_objects[0], nx.classes.graph.Graph)
        else:
            self.assertTrue(nx_objects, nx.classes.graph.Graph)

    def test_pad_with_zeros(self, space, points):
        padded_points = space.pad_with_zeros(points)

        expected_shape = (space.n_nodes, space.n_nodes)

        if type(points) is list:
            for point in padded_points:
                self.assertEqual(point.adj.shape, expected_shape)

            self.assertEqual(len(padded_points), len(points))

        elif type(points) is self.testing_data._Point:
            self.assertEqual(padded_points.adj.shape, expected_shape)
            self.assertTrue(isinstance(padded_points, self.testing_data._Point))

        else:
            self.assertEqual(padded_points.shape[-2:], expected_shape)
            self.assertEqual(padded_points.ndim, points.ndim)
            if points.ndim == 3:
                self.assertEqual(padded_points.shape[0], points.shape[0])


class TestGraphPoint(PointTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
    testing_data = GraphTestData()

    def test_to_networkx(self, point):
        self.assertTrue(type(point.to_networkx()), nx.classes.graph.Graph)


class TestGraphSpaceMetric(PointSetMetricTestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP

    skip_test_geodesic_output_type = True

    testing_data = GraphSpaceMetricTestData()

    def test_dist_output_shape(self, dist_fnc, point_a, point_b):
        results = dist_fnc(point_a, point_b)

        n_dist = max(
            point_a.shape[0] if gs.ndim(point_a) > 2 else 1,
            point_b.shape[0] if gs.ndim(point_b) > 2 else 1,
        )
        self.assertTrue(results.size == n_dist)

    def test_geodesic(self, space_args, start_point, end_point, t, expected):

        space = self.testing_data._PointSet(*space_args)

        metric = self.testing_data._PointSetMetric(space)
        geodesic = metric.geodesic(start_point, end_point)
        pts_result = geodesic(t)
        self.assertAllClose(pts_result, expected)

    def test_geodesic_output_shape(self, metric, start_point, end_point, t):
        geodesic = metric.geodesic(start_point, end_point)

        is_multiple = gs.ndim(start_point) > 2 or gs.ndim(end_point) > 2
        n_geo = max(
            start_point.shape[0] if gs.ndim(start_point) > 2 else 1,
            end_point.shape[0] if gs.ndim(end_point) > 2 else 1,
        )
        n_t = len(t) if type(t) is list else 1
        d_array = 2

        results = geodesic(t)
        self.assertTrue(results.ndim == d_array + 1 + int(is_multiple))
        self.assertTrue(results.shape[-d_array - 1] == n_t)
        if is_multiple:
            self.assertTrue(results.shape[-d_array - 2] == n_geo)

    def test_geodesic_bounds(self, metric, start_point, end_point):
        geodesic = metric.geodesic(start_point, end_point)

        results = geodesic([0.0, 1.0])
        self.assertAllClose(results, gs.stack([start_point, end_point]))

    def test_align_point_to_point(self, metric, point_a, point_b, aligner, expected):
        permuted_point = metric.align_point_to_point(point_a, point_b)

        self.assertAllClose(permuted_point, expected)

    def test_align_point_to_geodesic(self, metric, geodesic, point, expected):
        aligned_point = metric.align_point_to_geodesic(geodesic, point)
        self.assertAllClose(aligned_point, expected)


class TestDecorators(TestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
    testing_data = DecoratorsTestData()

    def test_vectorize_graph(self, fnc, points):
        gs_array_type = type(gs.array([0]))
        vec_points = fnc(points)

        self.assertTrue(type(vec_points) is gs_array_type)
        if type(points) is gs_array_type:
            self.assertTrue(vec_points.shape == points.shape)
        else:
            n_points = len(points) if type(points) is list else 1
            if n_points > 1:
                self.assertTrue(vec_points.shape[0] == n_points)
            else:
                self.assertTrue(vec_points.ndim == 2)

    def test_vectorize_graph_to_points(self, fnc, points):
        vec_points = fnc(points)

        self.assertTrue(type(vec_points) is list)
        self.assertTrue(type(vec_points[0]) is self.testing_data._Point)

        if type(points) is list:
            n_points = len(points)
        elif type(points) is self.testing_data._Point:
            n_points = 1
        else:
            n_points = 1 if points.ndim == 2 else points.shape[0]

        self.assertTrue(len(vec_points) == n_points)


class TestAligner(TestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
    testing_data = AlignerTestData()

    def _is_same_equivalence_class(self, permuted_point, expected, dist_fnc):
        dist = dist_fnc(expected, permuted_point)
        self.assertAllClose(dist, 0.0)

    def test_align(
        self, metric, aligner, base_point, permute_point, expected, dist_fnc
    ):
        res = aligner.align(metric, base_point, permute_point)

        self._is_same_equivalence_class(res, expected, dist_fnc)

    def test_align_cmp_points(
        self, metric, aligner, base_point, permute_point, expected
    ):
        res = aligner.align(metric, base_point, permute_point)

        self.assertAllClose(res, expected)

    def test_align_output_shape(self, metric, aligner, base_point, permute_point):
        expected = aligner.align(metric, base_point, permute_point)

        is_multiple = (gs.ndim(base_point) > 2 and base_point.shape[0] > 1) or (
            gs.ndim(permute_point) > 2 and permute_point.shape[0] > 1
        )

        out_ndim = gs.ndim(expected)
        if is_multiple:
            n_out = max(
                base_point.shape[0] if gs.ndim(base_point) > 2 else 1,
                permute_point.shape[0] if gs.ndim(permute_point) > 2 else 1,
            )
            self.assertEqual(out_ndim, 3)
            self.assertEqual(expected.shape[0], n_out)
        else:
            self.assertEqual(out_ndim, 2)


class TestPointToGeodesicAligner(TestCase, metaclass=Parametrizer):
    skip_all = IS_NOT_NP
    testing_data = PointToGeodesicAlignerTestData()

    def test_align(self, aligner, geodesic, point, expected):
        aligned_point = aligner.align(geodesic, point)
        self.assertAllClose(aligned_point, expected)

    def test_dist(self, aligner, geodesic, point, expected, atol):
        dist = aligner.dist(geodesic, point)

        # TODO: make output type and shape consistent
        # self.assertTrue(
        #     type(dist) is type(expected),
        #     f"{type(dist)} is not {type(expected)} (input shape: {point.shape})"
        # )
        self.assertAllClose(dist, expected, atol=atol)
