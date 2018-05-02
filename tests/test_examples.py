"""Unit tests for the examples."""

import unittest

import examples.plot_geodesics_h2 as plot_geodesics_h2
import examples.plot_geodesics_s2 as plot_geodesics_s2
import examples.plot_geodesics_se3 as plot_geodesics_se3
import examples.plot_geodesics_so3 as plot_geodesics_so3
import examples.plot_grid_h2 as plot_grid_h2
import examples.tangent_pca_so3 as tangent_pca_so3


class TestExamples(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_plot_geodesics_h2(self):
        plot_geodesics_h2.main()

    def test_plot_geodesics_s2(self):
        plot_geodesics_s2.main()

    def test_plot_geodesics_se3(self):
        plot_geodesics_se3.main()

    def test_plot_geodesics_so3(self):
        plot_geodesics_so3.main()

    def test_plot_grid_h2(self):
        plot_grid_h2.main()

    def test_tangent_pca_so3(self):
        tangent_pca_so3.main()


if __name__ == '__main__':
        unittest.main()
