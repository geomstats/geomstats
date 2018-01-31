"""Unit tests for visualization module."""

import geomstats.visualization as visualization

import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestVisualizationMethods(unittest.TestCase):

    def test_trihedron_from_rigid_transformation(self):
        translation = np.array([1, 2, 3])
        rot_vec = np.array([-1, 3, 6])
        transfo = np.concatenate([rot_vec, translation])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = visualization.plot_points(transfo, ax)
        fig.savefig('test_plot_trihedron.png')

if __name__ == '__main__':
        unittest.main()
