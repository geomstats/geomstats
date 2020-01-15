from .__about__ import __version__  # NOQA

import geomstats.geometry.manifold
import geomstats.geometry.euclidean_space
import geomstats.geometry.hyperbolic_space
import geomstats.geometry.hypersphere
import geomstats.geometry.invariant_metric
import geomstats.geometry.lie_group
import geomstats.geometry.minkowski_space
import geomstats.geometry.spd_matrices_space
import geomstats.geometry.special_euclidean_group
import geomstats.geometry.special_orthogonal_group
# XXX: Why does flake8 complain about this not being used but is fine with the
#      other imports?
import geomstats.geometry.riemannian_metric  # NOQA
