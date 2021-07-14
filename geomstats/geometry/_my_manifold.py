"""Template file to illustrate how to create a manifold in geomstats.

For additional guidelines on how to contribute to geomstats, visit:
https://geomstats.github.io/contributing.html#contributing-code-workflow

Note: A manifold needs to be created with an associated test file.
The test file for this manifold can be found at:
tests/test__my_manifold.py.
"""
import geomstats.backend as gs
# Import the class(es) that MyManifold inherits from
from geomstats.geometry.manifold import Manifold


# This class inherits from the class Manifold.
# Inheritance in geomstats means that the class MyManifold will reuse code
# that is in the Manifold class.
class MyManifold(Manifold):
    r"""Give a one-liner description/definition of MyManifold.

    For example: Class for Euclidean spaces.

    Give a more detailed description/definition of MyManifold.
    For example: By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    List the parameters of MyManifold, i.e. the parameters given as inputs
    of the constructor __init__.

    For example:
    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    """

    def __init__(self, dim, another_parameter, **kwargs):
        super(MyManifold, self).__init__(dim)
        self.another_parameter = another_parameter

    # Implement the main methods of MyManifold, for example belongs:
    def belongs(self, point, atol=gs.atol):
        """Give a one-liner description of the method.

        For example: Evaluate if a point belongs to MyManifold.

        The signature of the method should match the signature of the parent
        method, in this case the method `belongs` from the class `Manifold`.

        List the parameters of the method.
        In what follows, the ellipsis ... indicate either nothing
        or any number n of elements, i.e. shape=[..., dim] means
        shape=[dim] or shape=[n, dim] for any n.
        All functions/methods of geomstats should work for any number
        of inputs. In the case of the method `belongs`, it means:
        for any number of input points.
        For example:

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Tolerance, unused.
            Optional, default: backend atol

        List the outputs of the method.
        For example:

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        # Perform operations to check if point belongs
        # to the manifold, for example:
        belongs = point.shape[-1] == self.dim
        if gs.ndim(point) == 2:
            belongs = gs.tile([belongs], (point.shape[0],))
        return belongs

    # Another example of method of MyManifold.
    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether vector is tangent to the manifold at base_point.

        In what follows, the ellipsis ... indicates either nothing
        or any number n of elements, i.e. shape=[..., dim] means
        shape=[dim] or shape=[n, dim] for any n.
        All functions/methods of geomstats should work for any number
        of inputs. In the case of the function `is_tangent`, it means:
        for any number of input vectors.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance threshold

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        # Perform operations to determine if vector is a tangent vector,
        # for example:
        is_tangent = gs.shape(vector)[-1] == self.dim
        if gs.ndim(vector) == 2:
            is_tangent = gs.tile([is_tangent], (vector.shape[0],))
        return is_tangent

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """

    def random_point(self, n_samples=1, bound=1.):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
