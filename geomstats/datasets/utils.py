"""Loading toy datasets."""

import json
import os

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


DATA_FOLDER = os.path.join(
    'geomstats', 'datasets', 'data')

CITIES_PATH = os.path.join(
    DATA_FOLDER, 'cities/cities.json')
POSES_PATH = os.path.join(
    DATA_FOLDER, 'poses/poses.json')
KARATE_PATH = os.path.join(
    DATA_FOLDER, 'graph_karate/karate.txt')
KARATE_LABELS_PATH = os.path.join(
    DATA_FOLDER, 'graph_karate/karate_labels.txt')
GRAPH_RANDOM_PATH = os.path.join(
    DATA_FOLDER, 'graph_random/graph_random.txt')
GRAPH_RANDOM_LABELS_PATH = os.path.join(
    DATA_FOLDER, 'graph_random/graph_random_labels.txt')


def load_cities():
    """Load data from data/cities/cities.json.

    Returns
    -------
    data : array-like, shape=[50, 2]
        Array with each row representing one sample,
        i. e. latitude and longitude of a city.
        Angles are in radians.
    name : list
        List of city names.
    """
    with open(CITIES_PATH, encoding='utf-8') as json_file:
        data_file = json.load(json_file)

        names = [row['city'] for row in data_file]
        data = list(map(
            lambda row: [row[
                col_name] / 180 * gs.pi for col_name in ['lat', 'lng']],
            data_file))

    data = gs.array(data)

    colat = gs.pi / 2 - data[:, 0]
    colat = gs.expand_dims(colat, axis=1)
    lng = gs.expand_dims(data[:, 1] + gs.pi, axis=1)

    data = gs.concatenate([colat, lng], axis=1)
    sphere = Hypersphere(dim=2)
    data = sphere.spherical_to_extrinsic(data)
    return data, names


def load_poses(only_rotations=True):
    """Load data from data/poses/poses.csv.

    Returns
    -------
    data : array-like, shape=[5, 3] or shape=[5, 6]
        Array with each row representing one sample,
        i. e. one 3D rotation or one 3D rotation + 3D translation.
    img_paths : list
        List of img paths.
    """
    data = []
    img_paths = []
    so3 = SpecialOrthogonal(n=3, point_type='vector')

    with open(POSES_PATH) as json_file:
        data_file = json.load(json_file)

        for row in data_file:
            pose_mat = gs.array(row['rot_mat'])
            pose_vec = so3.rotation_vector_from_matrix(pose_mat)
            if not only_rotations:
                trans_vec = gs.array(row['trans_mat'])
                pose_vec = gs.concatenate([pose_vec, trans_vec], axis=-1)
            data.append(pose_vec)
            img_paths.append(row['img'])

    data = gs.array(data)
    return data, img_paths
