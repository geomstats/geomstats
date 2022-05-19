"""Unit tests for Wrapped gaussian process."""

from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.wrapped_gaussian_process import WrappedGaussianProcess


class TestWrappedGaussianProcess(geomstats.tests.TestCase):
    def setup_method(self):
        gs.random.seed(1234)
        self.n_samples = 20

        # Set up for hypersphere
        self.dim_sphere = 2
        self.shape_sphere = (self.dim_sphere + 1,)
        self.sphere = Hypersphere(dim=self.dim_sphere)

        self.intercept_sphere_true = gs.array([0.0, -1.0, 0.0])
        self.coef_sphere_true = gs.array([1.0, 0.0, 0.5])

        # set up the prior
        self.prior = lambda x: self.sphere.metric.exp(
            x * self.coef_sphere_true,
            base_point=self.intercept_sphere_true,
        )

        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))

        # generate data
        X = gs.linspace(0.0, 1.5 * gs.pi, self.n_samples)
        self.X_sphere = gs.reshape((X - gs.mean(X)), (-1, 1))
        # generate the geodesic
        y = self.prior(self.X_sphere)
        # Then add orthogonal sinusoidal oscillations

        o = (1.0 / 20.0) * gs.array([-0.5, 0.0, 1.0])
        o = self.sphere.to_tangent(o, base_point=y)
        s = self.X_sphere * gs.sin(5.0 * gs.pi * self.X_sphere)
        self.y_sphere = self.sphere.metric.exp(s * o, base_point=y)

    def test_fit_hypersphere(self):
        """Test the fit method"""
        wgpr = WrappedGaussianProcess(
            self.sphere, metric=self.sphere.metric, prior=self.prior, kernel=self.kernel
        )
        wgpr.fit(self.X_sphere, self.y_sphere)
        self.assertAllClose(wgpr.score(self.X_sphere, self.y_sphere), 1)

    def test_predict_hypersphere(self):
        """Test the predict method"""
        wgpr = WrappedGaussianProcess(
            self.sphere, metric=self.sphere.metric, prior=self.prior, kernel=self.kernel
        )
        wgpr.fit(self.X_sphere, self.y_sphere)
        y, std = wgpr.predict(self.X_sphere, return_tangent_std=True)
        self.assertAllClose(std, gs.zeros(std.shape), atol=1e-4)
        self.assertAllClose(y, self.y_sphere, atol=1e-4)

    def test_samples_y_hypersphere(self):
        """Test the samples_y method"""
        wgpr = WrappedGaussianProcess(
            self.sphere, metric=self.sphere.metric, prior=self.prior, kernel=self.kernel
        )
        wgpr.fit(self.X_sphere, self.y_sphere)
        y = wgpr.sample_y(self.X_sphere, n_samples=100)
        y_ = gs.reshape(gs.transpose(y, [0, 2, 1]), (-1, y.shape[1]))
        self.assertTrue(gs.all(self.sphere.belongs(y_)))
