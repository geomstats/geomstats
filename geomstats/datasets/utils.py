"""Loading toy datasets.

Refer to notebook: `geomstats/notebooks/01_data_on_manifolds.ipynb`
to visualize these datasets.

Lead author: Nina Miolane.
"""

import csv
import json
import os

import pandas as pd

import geomstats.backend as gs
from geomstats.datasets.prepare_graph_data import Graph
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

MODULE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODULE_PATH, "data")
CITIES_PATH = os.path.join(DATA_PATH, "cities", "cities.json")
CONNECTOMES_PATH = os.path.join(DATA_PATH, "connectomes/train_FNC.csv")
CONNECTOMES_LABELS_PATH = os.path.join(DATA_PATH, "connectomes/train_labels.csv")

POSES_PATH = os.path.join(DATA_PATH, "poses", "poses.json")
KARATE_PATH = os.path.join(DATA_PATH, "graph_karate", "karate.txt")
KARATE_LABELS_PATH = os.path.join(DATA_PATH, "graph_karate", "karate_labels.txt")
GRAPH_RANDOM_PATH = os.path.join(DATA_PATH, "graph_random", "graph_random.txt")
GRAPH_RANDOM_LABELS_PATH = os.path.join(
    DATA_PATH, "graph_random", "graph_random_labels.txt"
)
LEAVES_PATH = os.path.join(DATA_PATH, "leaves", "leaves.csv")
EMG_PATH = os.path.join(DATA_PATH, "emg", "emg.csv")
OPTICAL_NERVES_PATH = os.path.join(DATA_PATH, "optical_nerves", "optical_nerves.txt")
HANDS_PATH = os.path.join(DATA_PATH, "hands", "hands.txt")
HANDS_LABELS_PATH = os.path.join(DATA_PATH, "hands", "labels.txt")
CELLS_PATH = os.path.join(DATA_PATH, "cells", "cells.txt")
CELL_LINES_PATH = os.path.join(DATA_PATH, "cells", "cell_lines.txt")
CELL_TREATMENTS_PATH = os.path.join(DATA_PATH, "cells", "treatments.txt")


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
    with open(CITIES_PATH, encoding="utf-8") as json_file:
        data_file = json.load(json_file)

        names = [row["city"] for row in data_file]
        data = list(
            map(
                lambda row: [
                    row[col_name] / 180 * gs.pi for col_name in ["lat", "lng"]
                ],
                data_file,
            )
        )

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
    so3 = SpecialOrthogonal(n=3, point_type="vector")

    with open(POSES_PATH) as json_file:
        data_file = json.load(json_file)

        for row in data_file:
            pose_mat = gs.array(row["rot_mat"])
            pose_vec = so3.rotation_vector_from_matrix(pose_mat)
            if not only_rotations:
                trans_vec = gs.array(row["trans_mat"])
                pose_vec = gs.concatenate([pose_vec, trans_vec], axis=-1)
            data.append(pose_vec)
            img_paths.append(row["img"])

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
    mat : array-like, shape=[86, {[28, 28], 378}
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
    data = gs.array([[float(value) for value in row[1:]] for row in data_list[1:]])

    with open(CONNECTOMES_LABELS_PATH) as csvfile:
        labels = list(csv.reader(csvfile))
    target = gs.array([int(row[1]) for row in labels[1:]])
    if as_vectors:
        return data, patient_id, target
    mat = SkewSymmetricMatrices(28).matrix_representation(data)
    mat = gs.eye(28) - gs.transpose(gs.tril(mat), (0, 2, 1))
    mat = 1.0 / 2.0 * (mat + gs.transpose(mat, (0, 2, 1)))

    return mat, patient_id, target


def load_leaves():
    """Load data from data/leaves/leaves.xlsx.

    Returns
    -------
    beta_param : array-like, shape=[172, 2]
        Beta parameters of the beta distributions fitted to each
        leaf orientation angle sample of 172 species of plants.
    distrib_type: array-like, shape=[172, ]
        Leaf orientation angle distribution type for each of the 172 species.
    """
    data = pd.read_csv(LEAVES_PATH, sep=";")
    beta_param = gs.array(data[["nu", "mu"]])
    distrib_type = gs.squeeze(gs.array(data["Distribution"]))
    return beta_param, distrib_type


def load_emg():
    """Load data from data/emg/emg.csv.

    Returns
    -------
    data_emg : pandas.DataFrame, shape=[731682, 10]
        Emg time serie for each of the 8 electrodes, with the time stamps
        and the label of the hand sign.
    """
    data_emg = pd.read_csv(EMG_PATH)
    return data_emg


def load_optical_nerves():
    """Load data from data/optical_nerves/optical_nerves.txt.

    Load the dataset of sets of 5 landmarks, labelled S, T, I, N, V, in 3D
    on monkeys' optical nerve heads:
    - 1st landmark (S): superior aspect of the retina,
    - 2nd landmark (T): side of the retina closest to the temporal
        bone of the skull,
    - 3rd landmark (N): nose side of the retina,
    - 4th landmark (I): inferior point,
    - 5th landmarks (V): optical nerve head deepest point.

    For each monkey, an experimental glaucoma was introduced in one eye,
    while the second eye was kept as control. This dataset can be used to
    investigate a significant difference between the glaucoma and the
    control eyes.

    Label 0 refers to a normal eye, and Label 1 to an eye with glaucoma.

    References
    ----------
        .. [PE2015] V. Patrangenaru and L. Ellingson. Nonparametric Statistics
          on Manifolds and Their Applications to Object Data, 2015.
          https://doi.org/10.1201/b18969


    Returns
    -------
    data : array-like, shape=[22, 5, 3]
        Data representing the 5 landmarks, in 3D, for 11 different monkeys.
    labels : array-like, shape=[22,]
        Labels in {0, 1} classifying the corresponding optical nerve as
        normal (label = 0) or glaucoma (label = 1).
    monkeys : array-like, shape=[22,]
        Indices in 0...10 referencing the index of the monkey to which a given
        optical nerve belongs.
    """
    nerves = pd.read_csv(OPTICAL_NERVES_PATH, sep="\t")
    nerves = nerves.set_index("Filename")
    nerves = nerves.drop(index=["laljn103.12b", "lalj0103.12b"])
    nerves = nerves.reset_index(drop=True)
    nerves_gs = gs.array(nerves.values)

    data = gs.reshape(nerves_gs, (nerves_gs.shape[0], -1, 3))
    labels = gs.tile([0, 1], [nerves_gs.shape[0] // 2])
    monkeys = gs.repeat(gs.arange(11), 2)

    return data, labels, monkeys


def load_hands():
    """Load data from data/hands/hands.txt and labels.txt.

    Load the dataset of hand poses, where a hand is represented as a
    set of 22 landmarks - the hands joints - in 3D.

    The hand poses represent two different hand poses:
    - Label 0: hand is in the position "Grab"
    - Label 1: hand is in the position "Expand"

    This is a subset of the SHREC 2017 dataset [SWVGLF2017].

    References
    ----------
        .. [SWVGLF2017] Q. De Smedt, H. Wannous, J.P. Vandeborre,
        J. Guerry, B. Le Saux, D. Filliat, SHREC'17 Track: 3D Hand Gesture
        Recognition Using a Depth and Skeletal Dataset, 10th Eurographics
        Workshop on 3D Object Retrieval, 2017.
        https://doi.org/10.2312/3dor.20171049


    Returns
    -------
    data : array-like, shape=[52, 22, 3]
        Hand data, represented as a list of 22 joints, specifically as
        the 3D coordinates of these joints.
    labels : array-like, shape=[52,]
        Label representing hands poses. Label 0: "Grab", Label 1: "Expand"
    bone_list : array-like
        List of bones, as a list of connexions between joints.
    """
    data = gs.array(pd.read_csv(HANDS_PATH, sep=" ").values)
    n_landmarks = 22
    dim = 3
    data = gs.reshape(data, (data.shape[0], n_landmarks, dim))
    labels = gs.array(pd.read_csv(HANDS_LABELS_PATH).values.squeeze())

    bone_list = gs.array(
        [
            [0, 1],
            [0, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [1, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [1, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [1, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [1, 18],
            [18, 19],
            [19, 20],
            [20, 21],
        ]
    )
    return data, labels, bone_list


def load_cells():
    """Load cell data.

    This cell dataset contains cell boundaries of mouse osteosarcoma
    (bone cancer) cells. The dlm8 cell line is derived from dunn and is more
    aggressive as a cancer. The cells have been treated with one of three
    treatments : control (no treatment), jasp (jasplakinolide)
    and cytd (cytochalasin D). These are drugs which perturb the cytoskelet
    of the cells.

    Returns
    -------
    cells : list of 650 planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order,
        their lengths are not necessarily equal.
    cell_lines : list of 650 strings
        List of the cell lines of each cell (dlm8 or dunn).
    treatments : list of 650 strings
        List of the treatments given to each cell (control, cytd or jasp).
    """
    with open(CELLS_PATH) as cells_file:
        cells = cells_file.read().split("\n\n")
    for i, cell in enumerate(cells):
        cell = cell.split("\n")
        curve = []
        for point in cell:
            coords = [int(coord) for coord in point.split()]
            curve.append(coords)
        cells[i] = gs.cast(gs.array(curve), gs.float32)
    with open(CELL_LINES_PATH) as cell_lines_file:
        cell_lines = cell_lines_file.read().split("\n")
    with open(CELL_TREATMENTS_PATH) as treatments_file:
        treatments = treatments_file.read().split("\n")
    return cells, cell_lines, treatments
