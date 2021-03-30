"""Template file to illustrate how to create a manifold in geomstats.

For additional guidelines on how to contribute to geomstats, visit:
https://geomstats.github.io/contributing.html#contributing-code-workflow

Note: A manifold needs to be created with an associated test file.
The test file for this manifold can be found at:
tests/test__my_manifold.py.
"""

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
    def belongs(self, point):
        """Give a one-liner description of the method.

        For example: Evaluate if a point belongs to MyManifold.

        List the parameters of the method.
        For example:

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.

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
        return belongs

    # Another example of method of MyManifold.
    def is_tangent(self, vector, base_point=None):
        """Check whether vector is tangent to the manifold at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        # Perform operations to determine if vector is a tangent vector,
        # for example:
        is_tangent = vector.shape[-1] == self.dim
        return is_tangent
