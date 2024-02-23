"""Group actions."""

import math
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.vectorization import get_batch_shape


class GroupAction(ABC):
    """Base class for group action."""

    @abstractmethod
    def __call__(self, group_elem, point):
        """Perform action of a group element on a manifold point.

        Parameters
        ----------
        group_elem : array-like
            The element of a group.
        point : array-like
            A point on a manifold.

        Returns
        -------
        orbit_point : array-like
            A point on the orbit of point.
        """


class CongruenceAction(GroupAction):
    """Congruence action."""

    def __call__(self, group_elem, point):
        """Congruence action of a group element on a matrix.

        Parameters
        ----------
        group_elem : array-like, shape=[..., n, n]
            The element of a group.
        point : array-like, shape=[..., n, n]
            A point on a manifold.

        Returns
        -------
        orbit_point : array-like, shape=[..., n, n]
            A point on the orbit of point.
        """
        return Matrices.mul(group_elem, point, Matrices.transpose(group_elem))


class PermutationAction(CongruenceAction):
    """Congruence action of the permutation group on matrices."""

    def __call__(self, group_elem, point):
        """Congruence action of a group element on a matrix.

        Parameters
        ----------
        group_elem : array-like, shape=[..., n]
            Permutations where in position i we have the value j meaning
            the node i should be permuted with node j.
        point : array-like, shape=[..., n, n]
            A point on a manifold.

        Returns
        -------
        orbit_point : array-like, shape=[..., n, n]
            A point on the orbit of point.
        """
        perm_mat = permutation_matrix_from_vector(group_elem, dtype=point.dtype)
        return super().__call__(perm_mat, point)


class RowPermutationAction(GroupAction):
    """Action of the permutation group on matrices by multiplication."""

    def __call__(self, group_elem, point):
        """Permutation action applied to matrix.

        Parameters
        ----------
        group_elem: array-like, shape=[..., n]
            Permutations where in position i we have the value j meaning
            the node i should be permuted with node j.
        point : array-like, shape=[..., n, n]
            Matrices to be permuted.

        Returns
        -------
        permuted_point : array-like, shape=[..., n, n]
            Permuted matrices.
        """
        perm_mat = permutation_matrix_from_vector(group_elem, dtype=point.dtype)
        return gs.matmul(Matrices.transpose(perm_mat), point)


def permutation_matrix_from_vector(group_elem, dtype=gs.int64):
    """Transform a permutation vector into a matrix."""
    batch_shape = get_batch_shape(1, group_elem)
    n = group_elem.shape[-1]

    if batch_shape:
        indices = gs.array(
            [
                (k, i, j)
                for (k, group_elem_) in enumerate(group_elem)
                for (i, j) in zip(range(n), group_elem_)
            ]
        )
    else:
        indices = gs.array(list(zip(range(n), group_elem)))

    return gs.array_from_sparse(
        data=gs.ones(math.prod(batch_shape) * n, dtype=dtype),
        indices=indices,
        target_shape=batch_shape + (n, n),
    )
