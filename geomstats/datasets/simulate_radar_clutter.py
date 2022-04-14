"""Simulate radar clutter.

Lead author: Yann Cabanes.
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh, toeplitz


def WeibullRNG(scale, shape, noOfRandomNumbers):
    """
    f(x) = shape / scale(x / scale) ^ (shape - 1) exp(- (x / scale) ^ shape) )
    E(x) = scale * gamma(1 + 1 / shape)
    median(x) = scale * (ln(2)) ^ (1 / shape)
    """
    WeibullRandomNumbers = scale * (-np.log(1 - np.random.rand(1, noOfRandomNumbers))) ** (1 / shape)
    return WeibullRandomNumbers


def gaussian_correlation_vector(t, mean=0, variance=1):
    """
    Sf(x) = 1 / (2 * pi * sigma**2)**(1/2) * exp(-(x - m)**2 / (2 * sigma**2))
    """
    correlation_vector = np.exp(
        1j * 2 * np.pi * mean * t) * np.exp(- 2 * np.pi ** 2 * variance * t ** 2)
    return correlation_vector


def triangular_correlation_vector(t, mean=0, variance=1):
    """
    Sf(x) = tri(a * (x - m)),
    where tri(x) = np.max(1 - np.abs(x), 0)
    """
    a_sinc = 1 / (6 * variance) ** (1/3)
    correlation_vector = np.exp(
        1j * 2 * np.pi * mean * t) * (1 / a_sinc) * np.sinc(t / a_sinc) ** 2
    return correlation_vector


def independent_correlation_vector(t, mean=0, variance=1):
    """Correlation vector for independent data."""
    n_samples = t.shape[0]
    correlation_vector = np.zeros([n_samples], dtype=complex)
    correlation_vector[0] = 1
    return correlation_vector


def exp_abs_correlation_vector(t, mean=0, variance=1):
    """Sf(x) = exp(-a * abs(x - m))"""
    a_abs = (4 / variance) ** (1/3)
    correlation_vector = np.exp(
        1j * 2 * np.pi * mean * t) * (2 * a_abs) / (a_abs ** 2 + 4 * np.pi ** 2 * t ** 2)
    return correlation_vector


def geometric_correlation_vector(t, mean=0, variance=1/2):
    """
    Sf(x) = (2 * a) / (a ** 2 + 4 * pi ** 2 * x ** 2)
    This function has no variance.
    Hence parameter variance does not correspond to a variance.
    """
    # a_abs = variance / (1 + variance)
    if variance <= 0 or variance >= 1:
        raise ValueError(
            'The parameter should be positive and lower than one.')
    # correlation_vector = np.exp(
    #     1j * 2 * np.pi * mean * t) * np.exp(- a_abs * (np.abs(t)))
    correlation_vector = np.exp(
        1j * 2 * np.pi * mean * t) * variance ** (np.abs(t))
    return correlation_vector


SHAPES_DICT = {
    'Gaussian': gaussian_correlation_vector,
    'triangular': triangular_correlation_vector,
    'independent': independent_correlation_vector,
    'exp_abs': exp_abs_correlation_vector,
    'geometric': geometric_correlation_vector
    }


def simulate_clutter(
        n_cells=1000,
        n_pulses=20,
        thermal_noise_power=1,
        add_texture=True,
        clutter_power=1e4,
        spectrum_shape='Gaussian',
        spectrum_mean=0,
        spectrum_variance=1,
        spatial_correlation_shape='independent',
        spatial_correlation_coefficient=0.5):

    np.random.seed()

    square_root_clutter_power = clutter_power ** 0.5

    """
    Simulate the texture.

    Parameter of "shape" b
    parameter of "scale" a
    E(x) = a * Gamma(1 + 1 / b)

    clutter = sqrt(t) X where X is Gaussian of covariance R(i, i) = 1
    and t follows a Weibull law.
    the clutter power = E(t)

    if no texture: texture = np.ones([1, n_cells])
    """

    """Compute the temporal autocorrelation matrix."""

    if spectrum_shape in SHAPES_DICT:
        temporal_correlation_function = SHAPES_DICT[spectrum_shape]
    else:
        temporal_correlation_function = independent_correlation_vector
        print(
            'Unknown spatial correlation.\n'
            'We assume that there is no spatial correlation.')
        
    t = np.linspace(0, n_pulses, num=n_pulses, endpoint=False)
    temporal_correlation_vector = temporal_correlation_function(t, mean=spectrum_mean, variance=spectrum_variance)
    temporal_correlation_matrix = toeplitz(temporal_correlation_vector)
    eigenvalues, eigenvectors = eigh(temporal_correlation_matrix)
    square_root_eigenvalues = eigenvalues ** 0.5
    square_root_temporal_correlation_matrix = eigenvectors @ np.diag(square_root_eigenvalues) @ eigenvectors.conj().T

    """Compute the spatial autocorrelation matrix."""

    if spatial_correlation_shape in SHAPES_DICT:
        spatial_correlation_function = SHAPES_DICT[spatial_correlation_shape]
    else:
        spatial_correlation_function = independent_correlation_vector

    t = np.linspace(0, n_cells, num=n_cells, endpoint=False)
    spatial_correlation_vector = spatial_correlation_function(
        t,
        mean=0,
        variance=spatial_correlation_coefficient)
    spatial_correlation_matrix = toeplitz(spatial_correlation_vector)
    eigenvalues, eigenvectors = eigh(spatial_correlation_matrix)
    square_root_eigenvalues = eigenvalues ** 0.5
    square_root_spatial_correlation_matrix = eigenvectors @ np.diag(square_root_eigenvalues) @ eigenvectors.conj().T

    """Simulate gaussian random matrix."""
    gaussian_matrix = (np.random.randn(n_cells, n_pulses) +
                       1j * np.random.randn(
                n_cells, n_pulses)) / np.sqrt(2)

    """Create the spatial and temporal correlations and add spatial textures."""
    dataset = square_root_spatial_correlation_matrix @ gaussian_matrix @ square_root_temporal_correlation_matrix.T

    """Add texture"""
    if add_texture:
        shape = 0.658
        scale = 0.7418
        texture = np.sqrt(WeibullRNG(scale, shape, n_cells))
        square_root_texture = texture ** 0.5
        dataset *= square_root_texture
    else:
        texture = np.ones([1, n_cells])

    """Add clutter power."""
    dataset *= square_root_clutter_power

    """Add thermal noise."""
    if thermal_noise_power > 0:
        dataset += np.sqrt(thermal_noise_power) * (np.random.randn(n_cells, n_pulses) + 1j * np.random.randn(n_cells, n_pulses)) / np.sqrt(2)

    return dataset, temporal_correlation_matrix, spatial_correlation_matrix, texture
