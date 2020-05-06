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
    data = []
    names = []

    with open(CITIES_PATH) as json_file:
        data_file = json.load(json_file)

        for row in data_file:
            lat_in_radians = row['lat'] / 90 * gs.pi / 2
            lng_in_radians = row['lng'] / 180 * gs.pi
            data.append(gs.array([lat_in_radians, lng_in_radians]))
            names.append(row['city'])

    data = gs.array(data)
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

    with open(POSES_PATH) as json_file:
        data_file = json.load(json_file)

        for row in data_file:
            pose_mat = gs.array(row['rot_mat'])
            if not only_rotations:
                trans_mat = gs.array(row['trans_mat'])
                trans_mat = gs.expand_dims(trans_mat, axis=1)
                pose_mat = gs.concatenate(
                    [pose_mat, trans_mat], axis=1)
                pose_mat = gs.concatenate(
                    [pose_mat, gs.array([[0., 0., 0., 1.]])],
                    axis=0)
            data.append(pose_mat)
            img_paths.append(row['img'])

    data = gs.array(data)
    return data, img_paths
