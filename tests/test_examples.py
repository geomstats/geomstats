"""Unit tests for the examples."""

import os
import unittest

import examples.plot_geodesics_se3 as plot_geodesics_se3


class TestEuclideanSpaceMethods(unittest.TestCase):
    def test_plot_geodesics_se3(self):
        plot_geodesics_se3.main()
        #os.system('python3 examples/plot_geodesics_se3.py')


if __name__ == '__main__':
        unittest.main()
