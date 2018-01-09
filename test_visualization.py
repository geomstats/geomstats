"""Unit tests for visualization module."""

import visualization

import numpy as np
import unittest


class TestVisualizationMethods(unittest.TestCase):

    def test_trihedron_from_rigid_transformation(self):
        translation = np.array([1, 2, 3])
        rot_vec = np.array([-1, 3, 6])
        transfo = np.concatenate([rot_vec, translation])

        trihedron = visualization.trihedron_from_rigid_transformation(transfo)
        fig = visualization.plot_trihedron(trihedron)
        fig.savefig('test_plot_trihedron.png')

if __name__ == '__main__':
        unittest.main()
