"""Helper data classes for the MDM illustration example on SPD matrices."""

import itertools

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import EigenSummary
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class DatasetSPD2D:
    """Sample 2D SPD dataset.

    Data is of shape [n_samples * n_classes, n_features];
    Labels are of shape [n_samples * n_classes, n_classes].

    Attributes
    ----------
    n_samples: int
        Number of samples per class.
    n_features: int
        Dimension of data.
    n_classes: int
        Number of classes.
    """

    def __init__(self, n_samples=100, n_features=2, n_classes=3):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

    def generate_sample_dataset(self):
        """Generate the dataset.

        Returns
        -------
        X : array-like,
           shape = [n_samples * n_classes, n_features, n_features]
            Data.
        y : array-like, shape = [n_samples * n_classes, n_classes]
            Labels.
        """
        X, y = self.setup_data()
        X, y = shuffle(X, y)
        return X, y

    def setup_data(self):
        """Generate the un-shuffled dataset.

        Returns
        -------
        X : array-like,
           shape = [n_samples * n_classes, n_features, n_features]
            Data.
        y : array-like, shape = [n_samples * n_classes, n_classes]
            Labels.
        """
        mean_covariance_eigenvalues = gs.random.uniform(
            0.1, 5., (self.n_classes, self.n_features))
        var = 1.
        base_rotations = SpecialOrthogonal(n=self.n_features).random_gaussian(
            gs.eye(self.n_features), var, n_samples=self.n_classes)
        var_rotations = gs.random.uniform(
            .5, .75, (self.n_classes))

        y = gs.zeros((self.n_classes * self.n_samples, self.n_classes))
        X = []
        for i in range(self.n_classes):
            value_x = self.make_data(
                base_rotations[i], gs.diag(
                    mean_covariance_eigenvalues[i]), var_rotations[i])
            value_y = 1
            idx_y = [(j, i) for j in range(i * self.n_samples, (i + 1) *
                                           self.n_samples)]
            y = gs.assignment(y, value_y, idx_y)
            X.append(value_x)
        return gs.concatenate(X, axis=0), y

    def make_data(self, eigenspace, eigenvalues, var):
        """Generate Gaussian data from mean matrix and variance.

        Parameters
        ----------
        eigenspace : array-like, shape = [n, n]
            Data eigenvectors.
        eigenvalues : array-like, shape = [n, n]
            Eigenvalues matrix (diagonal matrix).
        var : float
            Variance of the wanted distribution.

        Returns
        -------
        spd_data : array-like, shape = [n, n]
            Output data.
        """
        spd = SPDMatrices(n=self.n_features)
        eigensummary = EigenSummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian_rotation_orbit(
            eigensummary=eigensummary, var_rotations=var,
            n_samples=self.n_samples)
        return spd_data

    def make_data_noisy(self, eigenspace, eigenvalues, var, var_eigenvalues):
        """Generate noisy Gaussian data from mean matrix and variance.

        Parameters
        ----------
        eigenspace : array-like, shape = [n, n]
            Data eigenvectors.
        eigenvalues : array-like, shape = [n, n]
            Eigenvalues matrix (diagonal matrix).
        var : float
            Variance of the wanted distribution.
        var_eigenvalues : float
            Noise within the distribution.

        Returns
        -------
        spd_data : array-like, shape = [n, n]
            Output data.
        """
        spd = SPDMatrices(n=self.n_features)
        eigensummary = EigenSummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian_rotation_orbit_noisy(
            eigensummary=eigensummary, var_rotations=var,
            var_eigenvalues=var_eigenvalues, n_samples=self.n_samples)
        return spd_data


def shuffle(X, y):
    """Shuffle the dataset.

    Parameters
    ----------
    X : array-like,
       shape = [n_samples * n_classes, n_features, n_features]
        Data to shuffle.
    y : array-like, shape = [n_samples * n_classes, n_classes]
        Labels to shuffle along with the data.

    Returns
    -------
    X_ : Shuffled version of X
    Y_ : Shuffled version of Y
    """
    idx_shuffled = gs.random.permutation(X.shape[0])
    X_ = X[idx_shuffled]
    y_ = y[idx_shuffled]

    is_tf = False
    if is_tf:
        product_idx_x = itertools.product(
            range(X.shape[0]), range(X.shape[1]), range(X.shape[2]))
        product_idx_y = itertools.product(
            range(y.shape[0]), range(y.shape[1]))
        idx_x = [(int(idx_shuffled[i]), j, k) for i, j, k in product_idx_x]
        idx_y = [(int(idx_shuffled[i]), j) for i, j in product_idx_y]
        X_ = gs.assignment(X, X, idx_x)
        y_ = gs.assignment(y, y, idx_y)

    return X_, y_


def get_label_at_index(i, labels):
    """Get the label of data point indexed by 'i'.

    Parameters
    ----------
    i : int
        Index of data point.
    labels : array-like, shape = [n_samples * n_classes, n_features]
        All labels.

    Returns
    -------
    label_i : int
        Class index.
    """
    label_i = gs.where(labels[i])[0][0]
    return label_i
