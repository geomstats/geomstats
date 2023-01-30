"""Unit tests for visualization."""

import geomstats.backend as gs
import matplotlib
import tests.conftest
from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.visualization.discrete_curves import DiscreteCurveViz

matplotlib.use("Agg")  # NOQA


class TestVisualizationDiscreteCurves(tests.conftest.TestCase):
    """Sjgnfkjgfdnglk."""

    def setup_method(self):
        """Sjgnfkjgfdnglk."""
        self.dc = DiscreteCurves(ambient_manifold=Euclidean(dim=3))
        self.linestyles = ["o-b", "o-r", "o-g"]
        self.labels = ["x", "y", "z"]
        self.title = "Set of discrete curves"
        self.n_times = 6  # number of intermediate geodesic curves
        self.index_a = 0  # index of curve A
        self.index_b = 1  # index of curve B
        self.parametrized_curve_a = lambda x: gs.transpose(
            gs.array([4 + 1 * gs.cos(gs.pi * x), 1 + 1 * gs.sin(gs.pi * x), 0 * x])
        )
        self.parametrized_curve_b = lambda x: gs.transpose(
            gs.array([0 * x, 5 + 1 * gs.cos(gs.pi * x), 1 + 1 * gs.sin(gs.pi * x)])
        )
        self.n_sampling_points = 6
        self.sampling_points = gs.linspace(0.0, 1, self.n_sampling_points + 1)
        self.dc_viz1 = DiscreteCurveViz(
            self.dc,
            [self.parametrized_curve_a, self.parametrized_curve_b],
            [self.sampling_points, self.sampling_points],
        )
        self.adjusted_sampling_points = self.sampling_points
        self.view_init = None
        self.sample_point_index = 2

    def test_set_curves(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.set_curves()

    def test_resample(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.resample(self.adjusted_sampling_points)

    def test_plot_3Dcurves(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.plot_3Dcurves(
            linestyles=self.linestyles, labels=self.labels, title=self.title
        )

    def test_plot_geodesic(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.plot_geodesic(
            n_times=self.n_times,
            inital_index=self.index_a,
            end_index=self.index_b,
            linestyles=self.linestyles,
            labels=self.labels,
            title=self.title,
        )

    def test_plot_geodesic_net(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.plot_geodesic_net(
            n_times=self.n_times,
            inital_index=self.index_a,
            end_index=self.index_b,
            linestyles=self.linestyles,
            labels=self.labels,
            title=self.title,
            view_init=self.view_init,
        )

    def test_plot_parallel_transport(self):
        """Sjgnfkjgfdnglk."""
        self.dc_viz1.plot_parallel_transport(
            n_times=self.n_times,
            sampling_point_index=self.sample_point_index,
            inital_index=self.index_a,
            end_index=self.index_b,
            linestyles=self.linestyles,
            labels=self.labels,
            title=self.title,
            view_init=self.view_init,
        )
