import geomstats.backend as gs
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.vectorization import get_n_points


class DiscreteSurfacesTestCase(ManifoldTestCase):
    def test_vertex_areas(self, point, expected, atol):
        """Test vertex_areas.

        Vertex area is the area of all of the triangles who are in contact
        with a specific vertex, according to the formula:
        vertex_areas = 2 * sum_incident_areas / 3.0
        """
        res = self.space.vertex_areas(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_normals(self, point, expected, atol):
        """Test normals.

        Recall that the magnitude of each normal vector is equal to
        the area of the face it is normal to.

        We compare the abs value of each normal vector array because:
        note that the "normals" variable here calculates the normals
        as pointing out of the surface, but the way that normals()
        was constructed makes it so that the normal vector could be
        pointing into the surface or out of the surface, (so it could
        either be positive or negative). Because of this, we make all
        of the normal vectors to the cube positive for testing
        purposes.
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

        n_points = get_n_points(self.space, point)
        if n_points == 1:
            res = gs.expand_dims(res, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for res_, expected_ in zip(res, expected):
            for normal, normal_ in zip(res_, expected_):
                _check_normal(normal, normal_)

    def test_surface_one_forms_prod(self, point, expected):
        """Test surface one forms."""
        res = self.space.surface_one_forms(point)

        first_vec = res[..., :, 0, :]
        second_vec = res[..., :, 1, :]
        inner_prods = gs.einsum("...ni,...ni->...n", first_vec, second_vec)

        if gs.ndim(inner_prods) == 1:
            inner_prods = gs.expand_dims(inner_prods, axis=0)
            expected = gs.expand_dims(expected, axis=0)

        for inner_prods_, expected_ in zip(inner_prods, expected):
            self.assertTrue(gs.all([prod in expected_ for prod in inner_prods_]))

    def test_faces_area(self, point, expected, atol):
        """Test faces area."""
        res = self.space.face_areas(point=point)
        self.assertAllClose(res, expected, atol=atol)


class ElasticMetricTestCase(RiemannianMetricTestCase):
    def test_path_energy_is_positive(self, path, atol):
        energy = self.space.metric.path_energy(path)
        self.assertTrue(gs.all(energy > -1 * atol))

    def test_path_energy_per_time_is_positive(self, path, atol):
        energy = self.space.metric.path_energy_per_time(path)
        self.assertTrue(gs.all(energy > -1 * atol))
