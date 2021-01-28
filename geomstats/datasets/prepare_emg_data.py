"""Pre-process time series into batched covariance matrices.

The user defines the number of time steps of the batches.
It starts by removing the transient signal by taking a margin on each side
of the sign change. It then creates batches of data that will be used to
build the covariance matrices. In practice, one needs to choose the size
of the batches big enough to get enough information, and small enough so
that the online classifier is reactive enough.
"""

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class TimeSeriesCovariance:
    """Class for generating a list of covariance matrices from time series.

    Parameters
    ----------
    time_serie : pandas.DataFrame
        Data contaning the time series for each electrodes, as well as
        the corresponding labels, and experiments.
    n_steps : int
        Size of the batches.
    n_elec : int
        The number of electrodes used for the recording.
    label_map : dictionary
        Encode the label into digits.
    margin : int
        Number of index to remove before and after a sign change (Can
        help getting a stationary signal).
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
    batches : array
        The start indexes of the batches to use to compute covariance matrices.
    margin : int
        Number of index to remove before and after a sign change (Can
        help getting a stationary signal).
    covs : array
        The covariance matrices.
    labels : array
        The digit labels corresponding to each batch.
    covec : array
        The vectorized version of the covariance matrices.
    diags : array
        The covariance matrices diagonals.
    """

    def __init__(self, data, n_steps, n_elec, label_map, margin=0):
        self.label_map = label_map
        self.data = data
        self.n_steps = n_steps
        self.n_elec = n_elec
        self.batches = np.array([])
        self.margin = margin
        self.covs = gs.array([])
        self.labels = gs.array([])
        self.covecs = gs.array([])
        self.diags = gs.array([])

    def _format_labels(self):
        """Convert the labels into digits."""
        self.data['y'] = self.data.label.map(lambda x: self.label_map[x])

    def _create_batches(self):
        """Create the batches used to compute covariance matrices.

        If margin != 0, we add an index margin at each label change
        to get stationary signal corresponding to each label.
        """
        start_ids = np.where(np.diff(self.data.y) != 0)[0]
        end_ids = np.append(start_ids[1:], len(self.data)) - self.margin
        start_ids += self.margin
        batches_list = [range(start_id, end_id - self.n_steps, self.n_steps)
                        for start_id, end_id in zip(start_ids, end_ids)]
        self.batches = gs.concatenate(batches_list)

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
        self.labels = gs.array(self.data.y.iloc[self.batches])
        self.covs = gs.array(covs)
        self.covecs = gs.array([SymmetricMatrices.to_vector(cov)
                                for cov in self.covs])
        self.diags = self.covs.diagonal(0, 1, 2)
