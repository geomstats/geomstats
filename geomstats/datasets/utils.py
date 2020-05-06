"""Loading toy datasets."""

import json
import os

import geomstats.backend as gs


DATA_FOLDER = os.path.join(
    'geomstats', 'datasets', 'data')

CITIES_PATH = 'cities/cities.json'
POSES_PATH = 'poses/poses.json'
KARATE_PATH = 'graph_karate/karate.txt'
KARATE_LABELS_PATH = 'graph_karate/karate_labels.txt'


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

    path = os.path.join(DATA_FOLDER, CITIES_PATH)
    data = gs.empty((n_samples, dim))
    names = []

    with open(path) as json_file:
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

    path = os.path.join(DATA_FOLDER, POSES_PATH)
    data = gs.empty((n_samples,) + shape)
    img_paths = []

    with open(path) as json_file:
        data_file = json.load(json_file)

        for i, row_i in enumerate(data_file):
            # rot_mat = gs.array([
            #     [row_i[1], row_i[2], row_i[3]],
            #     [row_i[4], row_i[5], row_i[6]],
            #     [row_i[10], row_i[11], row_i[12]]], dtype=gs.float64)
            rot_mat = row_i['rot_mat']
            data[i] = rot_mat
            img_paths.append(row_i['img'])

    return data, img_paths
