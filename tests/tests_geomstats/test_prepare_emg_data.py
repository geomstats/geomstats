"""Unit tests for TimeSeriesCovariance class."""

import geomstats.backend as gs
import tests.conftest
from geomstats.datasets.prepare_emg_data import TimeSeriesCovariance
from geomstats.datasets.utils import load_emg


@tests.conftest.np_and_autograd_only
class TestPrepareEmgData(tests.conftest.TestCase):
    """Class for testing the covariance creation from time series."""

    def setup_method(self):
        """Set up function."""
        self.n_steps = 100
        self.n_elec = 8
        self.label_map = {"rock": 0, "scissors": 1, "paper": 2, "ok": 3}
        self.margin = 1000
        data = load_emg()
        data = data[data.label != "rest"]
        self.emg_data = {
            "time_vec": gs.array(data.time),
            "raw_data": gs.array(data[[f"c{i}" for i in range(8)]]),
            "label": gs.array(data.label),
            "exp": gs.array(data.exp),
        }

        self.cov_transformer = TimeSeriesCovariance(
            self.emg_data, self.n_steps, self.n_elec, self.label_map, self.margin
        )
        self.cov_transformer.transform()

    def test_covariance_shape(self):
        """Test the shape of the covariance matrices."""
        result_shape = (len(self.cov_transformer.batches), self.n_elec, self.n_elec)
        self.assertTrue(self.cov_transformer.covs.shape == result_shape)

    def test_covec_shape(self):
        """Test the shape of the vectorized covariance."""
        dim_vec = int(self.n_elec * (self.n_elec + 1) / 2)
        result_shape = (len(self.cov_transformer.batches), dim_vec)
        shape = self.cov_transformer.covecs.shape
        self.assertTrue(shape == result_shape)

    def test_diag_shape(self):
        """Test the shape of the diagonal."""
        result_shape = (len(self.cov_transformer.batches), self.n_elec)
        shape = self.cov_transformer.diags.shape
        self.assertTrue(shape == result_shape)
