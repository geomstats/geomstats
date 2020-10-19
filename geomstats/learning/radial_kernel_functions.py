"""Radial kernel functions.

References
----------
https://en.wikipedia.org/wiki/Kernel_(statistics)
https://en.wikipedia.org/wiki/Radial_basis_function

Notes
-----
We chose not to apply the normalization coefficients used in some references
in order that the kernel functions integrate to 1 on the Euclidean space of
dimension 1.
"""

import geomstats.backend as gs


def _check_distance(distance):
    """Check if the distance if a non-negative real number."""
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    distance = gs.array(distance, dtype=float)
    return distance


def _check_bandwidth(bandwidth):
    """Check if the bandwidth is a positive real number."""
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    bandwidth = gs.array(bandwidth, dtype=float)
    return bandwidth


def uniform_radial_kernel(distance, bandwidth=1.0):
    """Uniform radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        gs.ones(distance.shape, dtype=float),
        gs.zeros(distance.shape, dtype=float))
    return weight


def triangular_radial_kernel(distance, bandwidth=1.0):
    """Triangular radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        1 - scaled_distance,
        gs.zeros(distance.shape, dtype=float))
    return weight


def parabolic_radial_kernel(distance, bandwidth=1.0):
    """Parabolic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        1 - scaled_distance ** 2,
        gs.zeros(distance.shape, dtype=float))
    return weight


def biweight_radial_kernel(distance, bandwidth=1.0):
    """Biweight radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        (1 - scaled_distance ** 2) ** 2,
        gs.zeros(distance.shape, dtype=float))
    return weight


def triweight_radial_kernel(distance, bandwidth=1.0):
    """Triweight radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        (1 - scaled_distance ** 2) ** 3,
        gs.zeros(distance.shape, dtype=float))
    return weight


def tricube_radial_kernel(distance, bandwidth=1.0):
    """Tricube radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        (1 - scaled_distance ** 3) ** 3,
        gs.zeros(distance.shape, dtype=float))
    return weight


def gaussian_radial_kernel(distance, bandwidth=1.0):
    """Gaussian radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    https://en.wikipedia.org/wiki/Radial_basis_function
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.exp(- scaled_distance ** 2 / 2)
    return weight


def cosine_radial_kernel(distance, bandwidth=1.0):
    """Cosine radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        gs.cos((gs.pi / 2) * scaled_distance),
        gs.zeros(distance.shape, dtype=float))
    return weight


def logistic_radial_kernel(distance, bandwidth=1.0):
    """Logistic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Kernel_(statistics)
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = 1 / (gs.exp(scaled_distance) + 2 + gs.exp(- scaled_distance))
    return weight


def sigmoid_radial_kernel(distance, bandwidth=1.0):
    """Sigmoid radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = 1 / (gs.exp(scaled_distance) + gs.exp(- scaled_distance))
    return weight


def bump_radial_kernel(distance, bandwidth=1.0):
    """Bump radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Radial_basis_function
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        gs.exp(- 1 / (1 - scaled_distance ** 2)),
        gs.zeros(distance.shape, dtype=float))
    return weight


def inverse_quadratic_radial_kernel(distance, bandwidth=1.0):
    """Inverse quadratic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Radial_basis_function
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = 1 / (1 + scaled_distance ** 2)
    return weight


def inverse_multiquadric_radial_kernel(distance, bandwidth=1.0):
    """Inverse multiquadric radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    References
    ----------
    https://en.wikipedia.org/wiki/Radial_basis_function
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = 1 / (1 + scaled_distance ** 2) ** (1 / 2)
    return weight


def laplacian_radial_kernel(distance, bandwidth=1.0):
    """Laplacian radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.

    Returns
    -------
    http://crsouza.com/2010/03/17/
    kernel-functions-for-machine-learning-applications/
    https://data-flair.training/blogs/svm-kernel-functions/
    """
    distance = _check_distance(distance)
    bandwidth = _check_bandwidth(bandwidth)
    scaled_distance = distance / bandwidth
    weight = gs.exp(- scaled_distance)
    return weight
