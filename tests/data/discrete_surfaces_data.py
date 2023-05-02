import random

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
from tests.data_generation import _ManifoldTestData

vertices, faces = data_utils.load_cube()
vertices = gs.array(vertices, dtype=gs.float64)
faces = gs.array(faces)


class DiscreteSurfacesTestData(_ManifoldTestData):
    space_args_list = [(faces,)]
    shape_list = [(8, 3)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = DiscreteSurfaces

    def vertex_areas_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def normals_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def surface_one_forms_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def faces_area_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)

    def surface_metric_matrices_test_data(self):
        smoke_data = [dict(faces=faces, point=vertices)]

        return self.generate_tests(smoke_data)
