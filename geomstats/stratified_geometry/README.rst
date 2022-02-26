Stratified spaces module
========================

The module ``stratified_geometry`` implements spaces that are length spaces but not
Riemannian manifolds. A length space `(M, d)` is a metric space where the distance between two
points of the space corresponds to the length of the shortest path between those two
points, where the length of the shortest path is measured with respect to `d`.
Every Riemannian manifold is also a length space.

Spaces that are length spaces but not Riemannian manifolds are mostly stratified spaces,
that is, a topological space that is the union of manifolds of possibly differing
dimension, equipped with a metric. It is the stratified nature of the space that makes
the use of a custom class ``Point`` preferable to the convention that points should be
array-like, also since the operations on ``Points`` are sometimes very non-linear.

This non-linearity is the reason why, for now, we have decided to furthermore break with
the compatibility to all backends, and we limit ourselves to the ``numpy`` backend.

The structure is designed as follows: the underlying set of points of a length space is
implemented in the class ``Space``, which takes as input also the class ``Point``, that
is the implementation of the - sometimes rather involved - elements of the space.
Finally, in line with the convention that the space and the geometry should be kept
separate for clarity for the user, the distance (we call metrics distance in order to
not get confused with the Riemannian metric) is implemented in a class ``Distance``.
This class can take optional arguments, for example different geometries of an ambient
space where the length space is embedded into.

As a small example, the tri-pod is already implemented.
