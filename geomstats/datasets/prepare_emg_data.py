"""Pre-process emg time series into batched covariance matrices.

The user defines the number of time steps of the batches.
"""

import numpy as np
import pandas as pd


def _vectorize_cov(cov_mat):
    """Vectorize a symmetric Matrix.

    Convert a symetric matrice of size n x n into a vector containing
    the n(n+1) lower elements (diagonal included).

    Parameters
    ----------
    cov_mat : np.array
        covariance matrix to vectorize.

    Returns
    -------
    covec : np.array
        vector of the n(n+1) lower elements of the matrix.
    """
    lcov = len(cov_mat)
    covec = []
    for i in range(lcov):
        for j in range(i + 1):
            covec.append(cov_mat[i][j])
    return np.array(covec)


class TimeSerieCovariance:
    """Class for generating a list of covariance matrix from time series.

    It starts by removing the transient signal by taking a margin on each side
    of the sign change. It then creates batches of data that will be used to
    build the covariance matrices. In practice, one needs to choose the size
    of the batches big enough to get enough information, and small enough so
    that the online classifier is reactive enough.

    Parameters
    ----------
    time_serie : pandas.DataFrame
        Data contaning the time series for each electrodes, as well as
        the corresponding labels, and experiments.
    n_steps : int
        Size of the batches.
    n_elec : int
        The number of electrodes used for the recording.

    Attributes
    ----------
    label_map : dictionary
        Encode the label into digits.
    data : pd.DataFrame
        Contains the raw time series data.
    n_steps : int
        Size of the batches.
    n_elec : int
        The number of electrodes used for the recording.
    batches : np.array
        The start indexes of the batches to use to compute covariance matrices.
    covs : np.array
        The covariance matrices in np.array format.
    df : pd.DataFrame
        Output data containing the covariance matrices along with the labels.
    """

    def __init__(self, data, n_steps, n_elec):
        self.label_map = {'rock': 0, 'scissors': 1, 'paper': 2, 'ok': 3}
        self.data = data
        self.n_steps = n_steps
        self.n_elec = n_elec
        self.batches = np.array([])
        self.covs = np.array([])
        self.df = pd.DataFrame()

    def _format_labels(self):
        """Remove the rest sign and converts the labels into digits."""
        self.data = self.data[self.data.label != 'rest']
        self.data['y'] = self.data.label.map(lambda x: self.label_map[x])

    def _create_batches(self):
        """Create the batches used to compute covariance matrices.

        Adding a time margin at each sign change to get stationary
        signal corresponding to each sign.
        """
        start_sign = np.where(np.diff(self.data.y) != 0)[0]
        end_sign = np.append(start_sign[1:], len(self.data)) - 1000
        start_sign += 1000
        self.batches = np.concatenate([range(start_sign[j],
                                             end_sign[j] - self.n_steps,
                                             self.n_steps)
                                      for j in range(len(start_sign))])

    def transform(self):
        """Transform the time serie into batched covariance matrices.

        We link the covariance matrices to its corresponding vectors,
        variance vector, labels, and experiments into a DataFrame.
        """
        self._format_labels()
        self._create_batches()
        covs = []
        for i in self.batches:
            x = self.data.iloc[i: i + self.n_steps, 1: 1 + self.n_elec].values
            covs.append(np.cov(x.transpose()))
        self.df['cov'] = covs
        self.df['label'] = list(self.data.y.iloc[self.batches])
        self.df['exp'] = list(self.data.exp.iloc[self.batches])
        self.covs = np.array(covs)
        self.df['covec'] = [_vectorize_cov(cov) for cov in self.covs]
        self.df['diag'] = list(self.covs.diagonal(0, 1, 2))
