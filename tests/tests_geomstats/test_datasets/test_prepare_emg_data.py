import pytest

import geomstats.backend as gs
from geomstats.datasets.prepare_emg_data import TimeSeriesCovariance
from geomstats.datasets.utils import load_emg
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase, np_and_autograd_only, pytorch_backend

from .data.prepare_emg_data import TimeSeriesCovarianceTestData


def _get_transformer():
    n_steps = 100
    n_elec = 8
    label_map = {"rock": 0, "scissors": 1, "paper": 2, "ok": 3}
    margin = 1000
    data = load_emg()
    data = data[data.label != "rest"]
    emg_data = {
        "time_vec": gs.array(data.time),
        "raw_data": gs.array(data[[f"c{i}" for i in range(8)]]),
        "label": gs.array(data.label),
        "exp": gs.array(data.exp),
    }

    return TimeSeriesCovariance(emg_data, n_steps, n_elec, label_map, margin)


@np_and_autograd_only
class TestTimeSeriesCovariance(TestCase, metaclass=DataBasedParametrizer):
    if not pytorch_backend():
        transformer = _get_transformer()
    testing_data = TimeSeriesCovarianceTestData()

    def setup_method(self):
        self.transformer.transform()

    @pytest.mark.shape
    def test_covariance_shape(self):
        """Test the shape of the covariance matrices."""
        n_elec = self.transformer.n_timeseries
        result_shape = (len(self.transformer.batches), n_elec, n_elec)
        self.assertTrue(self.transformer.covs.shape == result_shape)

    @pytest.mark.shape
    def test_covec_shape(self):
        """Test the shape of the vectorized covariance."""
        n_elec = self.transformer.n_timeseries
        dim_vec = int(n_elec * (n_elec + 1) / 2)
        result_shape = (len(self.transformer.batches), dim_vec)
        shape = self.transformer.covecs.shape
        self.assertTrue(shape == result_shape)

    @pytest.mark.shape
    def test_diag_shape(self):
        """Test the shape of the diagonal."""
        n_elec = self.transformer.n_timeseries
        result_shape = (len(self.transformer.batches), n_elec)
        shape = self.transformer.diags.shape
        self.assertTrue(shape == result_shape)
