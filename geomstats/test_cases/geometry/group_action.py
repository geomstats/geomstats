import itertools
import random

import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class PermutationGroup:
    """Permutation group.

    A basic (non-optimized) implementation of the permutation group with points
    represented as 1d arrays for testing purposes.
    """

    def __init__(self, dim):
        self.dim = dim
        self.shape = (dim,)
        self.identity = gs.arange(dim)

        self._all_perms = list(itertools.permutations(range(self.dim), self.dim))

    def random_point(self, n_samples=1):
        """Sample random point.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        random_point : array-like, shape=[..., dim]
        """
        random_point = gs.array(random.sample(self._all_perms, n_samples))
        if n_samples == 1:
            return random_point[0]

        return random_point

    def _inverse_single(self, group_elem):
        """Inverse element.

        Parameters
        ----------
        group_elem : array-like, shape=[dim]

        Returns
        -------
        inverse_group_elem : array-like, shape=[dim]
        """
        inverse_group_elem = gs.zeros_like(group_elem)
        for val, index in enumerate(group_elem):
            inverse_group_elem[index] = val
        return inverse_group_elem

    def inverse(self, group_elem):
        """Inverse element.

        Parameters
        ----------
        group_elem : array-like, shape=[..., dim]

        Returns
        -------
        inverse_group_elem : array-like, shape=[..., dim]
        """
        if group_elem.ndim == 1:
            return self._inverse_single(group_elem)

        return gs.stack(
            [self._inverse_single(group_elem_) for group_elem_ in group_elem]
        )


class GroupActionTestCase(TestCase):
    """Group action test case.

    Requires: `group_action`, `space`, `group`.
    """

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)
            self.group_data_generator = RandomDataGenerator(self.group)

    @pytest.mark.random
    def test_identity_action(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        orbit_point = self.group_action(self.group.identity, point)
        self.assertAllClose(point, orbit_point)

    def test_action(self, group_elem, point, expected, atol):
        orbit_point = self.group_action(group_elem, point)
        self.assertAllClose(orbit_point, expected, atol=atol)

    @pytest.mark.vec
    def test_action_vec(self, n_reps, atol):
        group_elem = self.group_data_generator.random_point()
        point = self.data_generator.random_point()

        expected = self.group_action(group_elem, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(group_elem=group_elem, point=point, expected=expected, atol=atol)
            ],
            arg_names=["group_elem", "point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inverse_action(self, n_points, atol):
        group_elem = self.group_data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        orbit_point = self.group_action(group_elem, point)
        point_ = self.group_action(self.group.inverse(group_elem), orbit_point)
        self.assertAllClose(point_, point, atol)
