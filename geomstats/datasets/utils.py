"""Loading toy datasets."""

import csv
import json
import os

import geomstats.backend as gs
from geomstats.datasets.prepare_graph_data import Graph
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


MODULE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_PATH, 'data')
CITIES_PATH = os.path.join(
    DATA_PATH, 'cities', 'cities.json')
CONNECTOMES_PATH = os.path.join(
    DATA_PATH, 'connectomes/train_FNC.csv')
CONNECTOMES_LABELS_PATH = os.path.join(
    DATA_PATH, 'connectomes/train_labels.csv')

POSES_PATH = os.path.join(
    DATA_PATH, 'poses', 'poses.json')
KARATE_PATH = os.path.join(
    DATA_PATH, 'graph_karate', 'karate.txt')
KARATE_LABELS_PATH = os.path.join(
    DATA_PATH, 'graph_karate', 'karate_labels.txt')
GRAPH_RANDOM_PATH = os.path.join(
    DATA_PATH, 'graph_random', 'graph_random.txt')
GRAPH_RANDOM_LABELS_PATH = os.path.join(
    DATA_PATH, 'graph_random', 'graph_random_labels.txt')


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


def load_random_graph():
    """Load data from data/graph_random.

    Returns
    -------
    graph: prepare_graph_data.Graph
        Graph containing nodes, edges, and labels from the random dataset.
    """
    return Graph(GRAPH_RANDOM_PATH, GRAPH_RANDOM_LABELS_PATH)


def load_karate_graph():
    """Load data from data/graph_karate.

    Returns
    -------
    graph: prepare_graph_data.Graph
        Graph containing nodes, edges, and labels from the karate dataset.
    """
    return Graph(KARATE_PATH, KARATE_LABELS_PATH)


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


def load_connectomes(as_vectors=False):
    """Load data from brain connectomes.

    Load the correlation data from the kaggle MSLP 2014 Schizophrenia
    Challenge. The original data came as flattened vectors, but if `raw=True`
    is passed, the correlation values are reshaped as symmetric matrices with
    ones on the diagonal.

    Parameters
    ----------
    as_vectors : bool
        Whether to return raw data as vectors or as symmetric matrices.
        Optional, default: False

    Returns
    -------
    data : array-like, shape=[86, {[28, 28], 378}
        Connectomes.
    patient_id : array-like, shape=[86,]
        Patient unique identifiers
    target : array-like, shape=[86,]
        Labels, whether patients belong to the diseased class (1) or control
        (0).
    """
    with open(CONNECTOMES_PATH) as csvfile:
        data_list = list(csv.reader(csvfile))
    patient_id = gs.array([int(row[0]) for row in data_list[1:]])
    data = gs.array(
        [[float(value) for value in row[1:]] for row in data_list[1:]])

    with open(CONNECTOMES_LABELS_PATH) as csvfile:
        labels = list(csv.reader(csvfile))
    target = gs.array([int(row[1]) for row in labels[1:]])
    if as_vectors:
        return data, patient_id, target
    mat = SkewSymmetricMatrices(28).matrix_representation(data)
    mat = gs.eye(28) - gs.transpose(gs.tril(mat), (0, 2, 1))
    mat = 1. / 2. * (mat + gs.transpose(mat, (0, 2, 1)))

    return mat, patient_id, target
