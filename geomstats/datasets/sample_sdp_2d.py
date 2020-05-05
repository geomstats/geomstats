"""Helper data classes for the MDM illustration example on SPD matrices."""

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class DatasetSPD_2D():
    """
    Sample 2D SPD dataset.

    Data is of shape [n_samples * n_classes, n_features];
    Labels are of shape [n_samples * n_classes, n_classes].
    """

    def __init__(self, n_samples=100, n_features=2, n_classes=3):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.data_helper = DataHelper()

    def generate_sample_dataset(self):
        """
        Generate the dataset.

        :return:
        X: array-like, shape = [n_samples * n_classes, n_features]: data
        Y: array-like, shape = [n_samples * n_classes, n_classes]: labels
        """
        X, Y = self.setup_data()
        X, Y = self.data_helper.shuffle(X, Y)
        return X, Y

    def setup_data(self):
        """
        Generate the un-shuffled dataset.

        :return:
        X: array-like, shape = [n_samples * n_classes, n_features]: data
        Y: array-like, shape = [n_samples * n_classes, n_classes]: labels
        """
        mean_covariance_eigenvalues = gs.random.uniform(
            0.1, 5., (self.n_classes, self.n_features))
        base_rotations = SpecialOrthogonal(n=self.n_features).random_gaussian(
            gs.eye(self.n_features), 1, n_samples=self.n_classes)
        # var_eigenvalues = gs.random.uniform(
        #     .04, .06, (self.n_classes, self.n_features))
        var_rotations = gs.random.uniform(
            .5, .75, (self.n_classes))

        # data
        cov = gs.zeros(
            (self.n_classes *
             self.n_samples,
             self.n_features,
             self.n_features))
        Y = gs.zeros((self.n_classes * self.n_samples, self.n_classes))
        for i in range(self.n_classes):
            cov[i * self.n_samples:(i + 1) * self.n_samples] = self.make_data(
                base_rotations[i], gs.diag(
                    mean_covariance_eigenvalues[i]), var_rotations[i])
            Y[i * self.n_samples:(i + 1) * self.n_samples, i] = 1
        return cov, Y

    def make_data(self, eigenspace, eigenvalues, var):
        """
        Generate Gaussian data from mean matrix and variance.

        :param eigenspace: array-like, shape = [n, n]
        :param eigenvalues: array-like, shape = [n, n] (diagonal matrix)
        :param var: float, variance of the wanted distribution.
        :return: array-like, shape = [n, n]: data.
        """
        spd = SPDMatrices(n=self.n_features)
        spd.set_eigensummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian(
            var_rotations=var, n_samples=self.n_samples)
        return spd_data

    def make_data_noisy(self, eigenspace, eigenvalues, var, var_eigenvalues):
        """
        Generate noisy Gaussian data from mean matrix and variance.

        :param eigenspace: array-like, shape = [n, n]
        :param eigenvalues: array-like, shape = [n, n] (diagonal matrix)
        :param var: float, variance of the wanted distribution.
        :param var_eigenvalues: float, noise within the distribution.
        :return: array-like, shape = [n, n]: data.
        """
        spd = SPDMatrices(n=self.n_features)
        spd.set_eigensummary(eigenspace, eigenvalues)
        spd_data = spd.random_gaussian_noisy(
            var_rotations=var, noise=var_eigenvalues, n_samples=self.n_samples)
        return spd_data


class DataHelper():
    """
    DataHelper provides simple functions to handle data.

    Data is assumed of the following shape:
    X: Data, shape=[n_samples, ...]
    Y: Labels, shape=[n_samples, n_classes] (one-hot encoding)
    """

    @staticmethod
    def shuffle(X, Y):
        """
        Shuffle the dataset.

        :param X: array-like, shape = [n_samples * n_classes, n_features]
        :param Y: array-like, shape = [n_samples * n_classes, n_classes]
        :return: Co-shuffled version of X and Y
        """
        tmp = list(zip(X, Y))
        gs.random.shuffle(tmp)
        X, Y = zip(*tmp)
        X = gs.array(X)
        Y = gs.array(Y)
        return X, Y

    @staticmethod
    def get_label_at_index(i, labels):
        """
        Get the label of data point indexed by 'i'.

        :param i: int, index of data point.
        :param labels: array-like, shape = [n_samples * n_classes, n_features]
        :return: int, class index.
        """
        return gs.where(labels[i])[0][0]
