"""Helper data classes for the MDM illustration example on SPD matrices."""

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import EigenSummary
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class DatasetSPD2D:
    """Sample 2D SPD dataset.

    Attributes
    ----------
    n_samples: int, number of samples per class;
    n_features: int, dimension of data;
    n_classes: int, number of classes;
    data_helper: DataHelper, wrapper for helper methods.

    Data is of shape [n_samples * n_classes, n_features];
    Labels are of shape [n_samples * n_classes, n_classes].
    """

    def __init__(self, n_samples=100, n_features=2, n_classes=3):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

    def generate_sample_dataset(self):
        """Generate the dataset.

        :return:
        X: array-like,
           shape = [n_samples * n_classes, n_features, n_features]: data
        y: array-like, shape = [n_samples * n_classes, n_classes]: labels
        """
        X, y = self.setup_data()
        X, y = shuffle(X, y)
        return X, y

    def setup_data(self):
        """Generate the un-shuffled dataset.

        :return:
        X: array-like,
           shape = [n_samples * n_classes, n_features, n_features]: data
        y: array-like, shape = [n_samples * n_classes, n_classes]: labels
        """
        mean_covariance_eigenvalues = gs.random.uniform(
            0.1, 5., (self.n_classes, self.n_features))
        base_rotations = SpecialOrthogonal(n=self.n_features).random_gaussian(
            gs.eye(self.n_features), 1, n_samples=self.n_classes)
        var_rotations = gs.random.uniform(
            .5, .75, (self.n_classes))

        X = gs.zeros(
            (self.n_classes *
             self.n_samples,
             self.n_features,
             self.n_features))
        y = gs.zeros((self.n_classes * self.n_samples, self.n_classes))
        for i in range(self.n_classes):
            X[i * self.n_samples:(i + 1) * self.n_samples] = self.make_data(
                base_rotations[i], gs.diag(
                    mean_covariance_eigenvalues[i]), var_rotations[i])
            y[i * self.n_samples:(i + 1) * self.n_samples, i] = 1
        return X, y

    def make_data(self, eigenspace, eigenvalues, var):
        """Generate Gaussian data from mean matrix and variance.

        :param eigenspace: array-like, shape = [n, n]
        :param eigenvalues: array-like, shape = [n, n] (diagonal matrix)
        :param var: float, variance of the wanted distribution.
        :return: array-like, shape = [n, n]: data.
        """
        spd = SPDMatrices(n=self.n_features)
        eigensummary = EigenSummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian_rotation_orbit(
            eigensummary=eigensummary, var_rotations=var,
            n_samples=self.n_samples)
        return spd_data

    def make_data_noisy(self, eigenspace, eigenvalues, var, var_eigenvalues):
        """Generate noisy Gaussian data from mean matrix and variance.

        :param eigenspace: array-like, shape = [n, n]
        :param eigenvalues: array-like, shape = [n, n] (diagonal matrix)
        :param var: float, variance of the wanted distribution.
        :param var_eigenvalues: float, noise within the distribution.
        :return: array-like, shape = [n, n]: data.
        """
        spd = SPDMatrices(n=self.n_features)
        eigensummary = EigenSummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian_rotation_orbit_noisy(
            eigensummary=eigensummary, var_rotations=var,
            var_eigenvalues=var_eigenvalues, n_samples=self.n_samples)
        return spd_data


def shuffle(X, Y):
    """Shuffle the dataset.

    X: array-like,
       shape = [n_samples * n_classes, n_features, n_features]: data
    :param Y: array-like, shape = [n_samples * n_classes, n_classes]
    :return: Co-shuffled version of X and Y
    """
    tmp = list(zip(X, Y))
    gs.random.shuffle(tmp)
    X, Y = zip(*tmp)
    X = gs.array(X)
    Y = gs.array(Y)
    return X, Y


def get_label_at_index(i, labels):
    """Get the label of data point indexed by 'i'.

    :param i: int, index of data point.
    :param labels: array-like, shape = [n_samples * n_classes, n_features]
    :return: int, class index.
    """
    return gs.where(labels[i])[0][0]
