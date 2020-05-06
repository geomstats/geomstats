"""Loading toy datasets."""

import json
import os

import geomstats.backend as gs


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
    n_samples = 50
    dim = 2

    data = gs.empty((n_samples, dim))
    names = []

    with open(CITIES_PATH) as json_file:
        data_file = json.load(json_file)

        for i, row_i in enumerate(data_file):
            lat_in_radians = row_i['lat'] / 90 * gs.pi / 2
            lng_in_radians = row_i['lng'] / 180 * gs.pi
            data[i] = gs.array([lat_in_radians, lng_in_radians])
            names.append(row_i['city'])

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
    n_samples = 5
    shape = (3, 3) if only_rotations else (4, 4)

    data = gs.empty((n_samples,) + shape)
    img_paths = []

    with open(POSES_PATH) as json_file:
        data_file = json.load(json_file)

        for i, row_i in enumerate(data_file):
            rot_mat = row_i['rot_mat']
            data[i] = rot_mat
            img_paths.append(row_i['img'])

    return data, img_paths
