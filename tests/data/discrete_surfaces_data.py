import random

import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
from tests.data_generation import _ManifoldTestData

_, faces = data_utils.load_cube()


class DiscreteSurfacesTestData(_ManifoldTestData):
    space_args_list = [(faces,)]
    shape_list = [(8, 3)]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = DiscreteSurfaces
