"""Import main modules."""

__version__ = "2.5.0"

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    # mypy error: Cannot determine type of '__GEOMSTATS_SETUP__'
    __GEOMSTATS_SETUP__  # type: ignore
except NameError:
    __GEOMSTATS_SETUP__ = False

if __GEOMSTATS_SETUP__:
    import sys

    sys.stderr.write(
        "Partial import of geomstats during build "
        "process as some run dependencie are missing.\n"
    )
    # We are not importing the rest of geomstats during the build
    # process, as it may not be compiled yet
else:
    import geomstats._backend
    import geomstats._logging
    import geomstats.geometry.discrete_curves
    import geomstats.geometry.euclidean
    import geomstats.geometry.hyperbolic
    import geomstats.geometry.hypersphere
    import geomstats.geometry.invariant_metric
    import geomstats.geometry.landmarks
    import geomstats.geometry.lie_algebra
    import geomstats.geometry.lie_group
    import geomstats.geometry.manifold
    import geomstats.geometry.minkowski
    import geomstats.geometry.poincare_polydisk
    import geomstats.geometry.product_manifold
    import geomstats.geometry.product_riemannian_metric
    import geomstats.geometry.riemannian_metric
    import geomstats.geometry.skew_symmetric_matrices
    import geomstats.geometry.spd_matrices
    import geomstats.geometry.special_euclidean
    import geomstats.geometry.special_orthogonal  # NOQA
