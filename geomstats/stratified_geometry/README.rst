Stratified spaces module
========================

The module ``stratified_geometry`` implements spaces that are length spaces but not
Riemannian manifolds. A length space `(M, d)` is a metric space where the distance
between two points of the space corresponds to the length of the shortest path between
those two points, where the length of the shortest path is measured with respect to `d`.
A Riemannian manifold is also a length space.

Spaces that are length spaces but not Riemannian manifolds are often stratified spaces,
that is, a topological space that is the union of manifolds of possibly various
dimension, equipped with a metric (a distance). It is the stratified nature of the space
that makes the use of a custom class ``Point`` preferable to the convention that points
should be array-like, also since the operations on ``Point``s are sometimes very
non-linear.

This non-linearity is the reason why, for now, we have decided to furthermore break with
the convention that compatibility to all backends is required; we limit ourselves to the
``numpy`` backend.

The structure is designed as follows: the underlying set of points of a length space is
implemented in the class ``PointSet``, which takes as input also the class ``Point`` via
the concept of generic classes, such that ``Point``s are the implementation of the
sometimes rather involved elements of the space.

Finally, since the space and the geometry are not cleanly separable anymore due to the
introduction of a class ``Point`` for elements of the space, the class ``LengthSpace``
implements length spaces, enabling the computation of a distance and possibly geodesics
(we call metrics distance in order to not get confused with the Riemannian metric).
This class can take optional arguments, for example different geometries of an ambient
space where the length space is embedded into.

As a small example, the tripod is already implemented.
