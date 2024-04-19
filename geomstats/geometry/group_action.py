"""Group actions."""

import math
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.vectorization import get_batch_shape


class GroupAction(ABC):
    """Base class for group action."""

    @abstractmethod
    def __call__(self, group_elem, point):
        """Perform action of a group element on a manifold point.

        Parameters
        ----------
        group_elem : array-like, shape=[..., *group.shape]
            The element of a group.
        point : array-like, shape=[..., *space.shape]
            A point on a manifold.

        Returns
        -------
        orbit_point : array-like
            A point on the orbit of point.
        """


class LieAlgebraBasedGroupAction(GroupAction):
    """Action of a group on itself by composition.

    Group elements are represented by the vector representation
    of Lie algebra elements. This is amenable to perform gradient-based
    optimizations on the Lie algebra.

    Parameters
    ----------
    group : LieGroup.
    """

    def __init__(self, group):
        self._group = group

    @property
    def group_elem_shape(self):
        """Shape of the group element representation."""
        return (self._group.lie_algebra.dim,)

    def __call__(self, group_elem, point):
        """Compose action of a group element on a point.

        Parameters
        ----------
        group_elem : array-like, shape=[..., *group_elem_shape]
            Group element represented in the Lie algebra as a vector.
        point : array-like, shape=[..., *group.shape]
            Point on the group.

        Returns
        -------
        orbit_point : array-like, shape=[..., *group.shape]
            A point in the orbit of point.
        """
        algebra_elt = self._group.lie_algebra.matrix_representation(group_elem)
        group_elt = self._group.exp(algebra_elt)
        return self._group.compose(point, group_elt)


class LieAlgebraBasedSpecialOrthogonalAction(LieAlgebraBasedGroupAction):
    """Action of the special orthogonal group.

    Group elements are represented by the vector representation
    of Lie algebra elements. This is amenable to perform gradient-based
    optimizations on the Lie algebra.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices : n x n.
    """

    def __init__(self, n):
        super().__init__(SpecialOrthogonal(n, equip=False))


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
    """Transform a permutation vector into a matrix.

    Parameters
    ----------
    group_elem : array-like, shape=[..., n]
        The element of a group.
    dtype: gs.dtype
        Array dtype

    Returns
    -------
    mat_group_element : array-like, shape=[..., n, n]
        Matrix representation of group element.
    """
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
