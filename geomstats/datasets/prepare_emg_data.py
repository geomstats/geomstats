"""Pre-process time series into batched covariance matrices.

The user defines the number of time steps of the batches.
It starts by removing the transient signal by taking a margin on each side
of the sign change. It then creates batches of data that will be used to
build the covariance matrices. In practice, one needs to choose the size
of the batches big enough to get enough information, and small enough so
that the online classifier is reactive enough.

Lead author: Marius Guerard.
"""

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class TimeSeriesCovariance:
    """Class for generating a list of covariance matrices from time series.

    Prepare a TimeSeriesCovariance Object from time series in dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary with 'time', 'raw_data', 'label' as key
        and the corresponding array as values.
    n_steps : int
        Size of the batches.
    n_timeseries : int
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
    data_dict : dict
        Dictionary with 'time', 'raw_data', 'label' as key
        and the corresponding array as values.
    n_steps : int
        Size of the batches.
    n_timeseries : int
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

    def __init__(self, data, n_steps, n_timeseries, label_map, margin=0):
        self.label_map = label_map
        self.data = data
        self.n_steps = n_steps
        self.n_timeseries = n_timeseries
        self.batches = gs.array([])
        self.margin = margin
        self.covs = gs.array([])
        self.labels = gs.array([])
        self.covecs = gs.array([])
        self.diags = gs.array([])

    def _format_labels(self):
        """Convert the labels into digits."""
        self.data["y"] = gs.array([self.label_map[x] for x in self.data["label"]])

    def _create_batches(self):
        """Create the batches used to compute covariance matrices.

        If margin != 0, we add an index margin at each label change
        to get stationary signal corresponding to each label.
        """
        start_ids = gs.where(np.diff(self.data["y"]) != 0)[0]
        end_ids = np.append(start_ids[1:], len(self.data)) - self.margin
        start_ids += self.margin
        batches_list = [
            range(start_id, end_id - self.n_steps, self.n_steps)
            for start_id, end_id in zip(start_ids, end_ids)
        ]
        self.batches = np.int_(gs.concatenate(batches_list))

    def transform(self):
        """Transform the time series into batched covariance matrices.

        We also compute the corresponding vectors, variance vector,
        labels, and experiments.
        """
        if "y" not in self.data.keys():
            self._format_labels()
        self._create_batches()
        covs = []
        for i in self.batches:
            x = self.data["raw_data"][i : i + self.n_steps]
            covs.append(np.cov(x.transpose()))
        self.labels = gs.array(self.data["y"][self.batches])
        self.covs = gs.array(covs)
        self.covecs = gs.array([SymmetricMatrices.to_vector(cov) for cov in self.covs])
        self.diags = self.covs.diagonal(0, 1, 2)
