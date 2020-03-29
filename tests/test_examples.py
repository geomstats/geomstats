"""Unit tests for the examples."""

import os
import sys
import warnings

import examples.empirical_frechet_mean_uncertainty_sn as empirical_frechet_mean_uncertainty_sn  # NOQA
import examples.gradient_descent_s2 as gradient_descent_s2
import examples.loss_and_gradient_se3 as loss_and_gradient_se3
import examples.loss_and_gradient_so3 as loss_and_gradient_so3
import examples.plot_geodesics_h2 as plot_geodesics_h2
import examples.plot_geodesics_poincare_polydisk as plot_geodesics_poincare_polydisk # NOQA
import examples.plot_geodesics_s2 as plot_geodesics_s2
import examples.plot_geodesics_se3 as plot_geodesics_se3
import examples.plot_geodesics_so3 as plot_geodesics_so3
import examples.plot_grid_h2 as plot_grid_h2
import examples.plot_kmeans_manifolds as plot_kmeans_manifolds
import examples.plot_online_kmeans_s1 as plot_online_kmeans_s1
import examples.plot_online_kmeans_s2 as plot_online_kmeans_s2
import examples.plot_pole_ladder_s2 as plot_pole_ladder_s2
import examples.plot_square_h2_klein_disk as plot_square_h2_klein_disk
import examples.plot_square_h2_poincare_disk as plot_square_h2_poincare_disk
import examples.plot_square_h2_poincare_half_plane as plot_square_h2_poincare_half_plane  # NOQA
import examples.tangent_pca_s2 as tangent_pca_s2
import examples.tangent_pca_so3 as tangent_pca_so3
import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import geomstats.tests


class TestExamples(geomstats.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.stdout = open(os.devnull, 'w')

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=UserWarning)
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.figure()

    @geomstats.tests.np_only
    def test_empirical_frechet_mean_uncertainty_sn(self):
        empirical_frechet_mean_uncertainty_sn.main()

    @geomstats.tests.np_only
    def test_gradient_descent_s2(self):
        gradient_descent_s2.main(max_iter=32, output_file=None)

    def test_loss_and_gradient_so3(self):
        loss_and_gradient_so3.main()

    def test_loss_and_gradient_se3(self):
        loss_and_gradient_se3.main()

    @geomstats.tests.np_only
    def test_plot_geodesics_h2(self):
        plot_geodesics_h2.main()

    @geomstats.tests.np_only
    def test_plot_geodesics_poincare_polydisk(self):
        plot_geodesics_poincare_polydisk.main()

    @geomstats.tests.np_only
    def test_plot_geodesics_s2(self):
        plot_geodesics_s2.main()

    @geomstats.tests.np_only
    def test_plot_geodesics_se3(self):
        plot_geodesics_se3.main()

    @geomstats.tests.np_only
    def test_plot_geodesics_so3(self):
        plot_geodesics_so3.main()

    @geomstats.tests.np_only
    def test_plot_grid_h2(self):
        plot_grid_h2.main()

    @geomstats.tests.np_only
    def test_plot_square_h2_square_poincare_disk(self):
        plot_square_h2_poincare_disk.main()

    @geomstats.tests.np_only
    def test_plot_square_h2_square_poincare_half_plane(self):
        plot_square_h2_poincare_half_plane.main()

    @geomstats.tests.np_only
    def test_plot_square_h2_square_klein_disk(self):
        plot_square_h2_klein_disk.main()

    @geomstats.tests.np_only
    def test_tangent_pca_s2(self):
        tangent_pca_s2.main()

    @geomstats.tests.np_only
    def test_tangent_pca_so3(self):
        tangent_pca_so3.main()

    @geomstats.tests.np_only
    def test_plot_kmeans_manifolds(self):
        plot_kmeans_manifolds.main()

    @geomstats.tests.np_only
    def test_plot_online_kmeans_s1(self):
        plot_online_kmeans_s1.main()

    @geomstats.tests.np_only
    def test_plot_online_kmeans_s2(self):
        plot_online_kmeans_s2.main()

    @geomstats.tests.np_only
    def test_plot_pole_ladder_s2(self):
        plot_pole_ladder_s2.main()
