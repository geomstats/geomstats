import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.vectorization import get_n_points


class SurfaceTestCase(TestCase):
    def test_face_areas(self, point, expected, atol=gs.atol):
        """Check face areas.

        Parameters
        ----------
        point : Surface
            Surface.
        expected : array-like, shape=[..., n_faces]
            Expected face areas.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(point.face_areas, expected, atol=atol)

    def test_face_normals(self, point, expected, atol=gs.atol):
        """Check face normals.

        Parameters
        ----------
        point : Surface
            Surface.
        expected : array-like, shape=[..., n_faces, 3]
            Expected face normals.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(point.face_normals, expected, atol=atol)

    def test_face_centroids(self, point, expected, atol=gs.atol):
        """Check face centroids.

        Parameters
        ----------
        point : Surface
            Surface.
        expected : array-like, shape=[..., n_faces, 3]
            Expected face centroids.
        atol : float
            Absolute tolerance.
        """
        self.assertAllClose(point.face_centroids, expected, atol=atol)


class DiscreteSurfacesTestCase(ManifoldTestCase):
    def test_vertex_areas(self, point, expected, atol=gs.atol):
        """Check vertex areas.

        The area of a vertex is two thirds of the sum of the areas of the
        faces incident to it.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface.
        expected : array-like, shape=[..., n_vertices]
            Expected vertex areas.
        atol : float
            Absolute tolerance.
        """
        res = self.space.vertex_areas(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normals(self, point, expected, atol=gs.atol):
        """Check normals, up to orientation.

        The norm of the normal of a face is its area. Normals may point in or
        out of the surface, so each normal is compared up to its sign.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface.
        expected : array-like, shape=[..., n_faces, 3]
            Expected normals, up to their sign.
        atol : float
            Absolute tolerance.
        """

        def _check_normal(normal, normal_):
            self.assertTrue(
                gs.any(
                    [
                        gs.allclose(normal, normal_, atol=atol),
                        gs.allclose(normal, -normal_, atol=atol),
                    ]
                )
            )

        res = self.space.normals(point)

        n_points = get_n_points(self.space.point_ndim, point)
        if n_points == 1:
            res = gs.expand_dims(res, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for res_, expected_ in zip(res, expected):
            for normal, normal_ in zip(res_, expected_):
                _check_normal(normal, normal_)

    def test_surface_one_forms_prod(self, point, expected):
        """Check the inner products between the one-forms of each face.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface.
        expected : array-like, shape=[..., n_values]
            Values the inner products can take.
        """
        res = self.space.surface_one_forms(point)

        first_vec = res[..., :, 0, :]
        second_vec = res[..., :, 1, :]
        inner_prods = gs.einsum("...ni,...ni->...n", first_vec, second_vec)

        if gs.ndim(inner_prods) == 1:
            inner_prods = gs.expand_dims(inner_prods, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for inner_prods_, expected_ in zip(inner_prods, expected):
            self.assertTrue(gs.all([prod in expected_ for prod in inner_prods_]))

    def test_faces_area(self, point, expected, atol=gs.atol):
        """Check face areas.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface.
        expected : array-like, shape=[..., n_faces]
            Expected face areas.
        atol : float
            Absolute tolerance.
        """
        res = self.space.face_areas(point=point)
        self.assertAllClose(res, expected, atol=atol)
