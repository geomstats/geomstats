"""Principal Nested Spheres.

Principal Nested Spheres (PNS) for recursive dimension reduction on spheres.
Based on the implementation from the RNA-Classification repository.

Lead author: Kaisar Dauletbeck, Benjamin Eltzner.
"""

import logging
import sys
from math import exp, log, sin

from scipy import stats
from scipy.integrate import quad
from scipy.optimize import leastsq, minimize
from sklearn.base import BaseEstimator, TransformerMixin

import geomstats.backend as gs


def _quad_safe(func, a, b):
    """Wrap scipy.integrate.quad with graceful fallback."""
    try:
        return quad(func, a, b)[0]
    except Exception:
        return quad(func, a, b, limit=200)[0]


def _normalization(rho, sigma, dim, euclidean=False):
    """Compute normalizing constant for the folded normal likelihood."""

    def density(r):
        scaled = r / sigma
        return exp(-0.5 * (scaled - rho) ** 2) + exp(-0.5 * (scaled + rho) ** 2)

    try:
        if not euclidean:
            integrand = lambda r: (sin(r) ** (dim - 1)) * density(r)
            integral = _quad_safe(integrand, 0.0, gs.pi)
        else:
            integrand = lambda r: (r ** (dim - 1)) * density(r)
            integral = _quad_safe(integrand, 0.0, (20.0 + rho) * sigma)
        return max(sys.float_info.min, integral)
    except Exception:
        return max(sys.float_info.min, gs.sqrt(2.0 * gs.pi) * sigma)


def _compare_likelihoods(radii, dim, verbose=False, euclidean=False):
    """Likelihood ratio test between great and small sphere fits.

    Parameters
    ----------
    radii : array-like
        Angular radii of the projected points.
    dim : int
        Intrinsic dimension of the subsphere (ambient dimension - 1).
    verbose : bool
        If True, print debugging information.
    euclidean : bool
        Use Euclidean approximation if True.

    Returns
    -------
    prefer_great : bool
        True if statistical test prefers a great sphere.
    """
    radii = gs.to_numpy(radii)
    if radii.ndim == 0:
        radii = gs.array([float(radii)])
    mean = radii.mean()
    std = radii.std()

    def likelihood(params):
        rho, sigma = params
        penalty = 0.0
        scale = 2.0 * rho * sigma / gs.pi
        if (scale > 1.0) and not euclidean:
            rho = 0.5 * gs.pi / sigma
            penalty = scale
        norm_const = _normalization(rho, sigma, dim, euclidean=euclidean)
        folded = log(norm_const) + 0.5 * ((radii / sigma) - rho) ** 2
        denominator = gs.log(1.0 + gs.exp(-2.0 * rho * radii / sigma))
        return folded.sum() - denominator.sum() + penalty

    def likelihood_null(sigma):
        if hasattr(sigma, "__len__"):
            sigma = sigma[0]
        return likelihood([1.0, float(sigma)])

    initial_rho = max(mean / std, 1.0) if std > 0 else 1.0
    initial_sigma = std if std > 0 else 1.0

    try:
        mle = minimize(
            likelihood,
            gs.array([initial_rho, initial_sigma]),
            method="L-BFGS-B",
            bounds=(
                (0.0, gs.pi * 1e3),
                (max(1e-3, 0.25 * initial_sigma), max(10 * initial_sigma, 1e-2)),
            ),
        ).x
    except Exception as err:
        if verbose:
            logging.warning(f"Likelihood optimization failed: {err}")
        return True

    if mle[0] < 1.0:
        return True

    try:
        mle_null = minimize(
            likelihood_null,
            gs.array([mle[1]]),
            method="L-BFGS-B",
            bounds=((max(mle[1], initial_sigma), 10.0 * max(std, mle[1], 1e-3)),),
        ).x
    except Exception as err:
        if verbose:
            logging.warning(f"Null likelihood optimization failed: {err}")
        return True

    try:
        null_val = likelihood_null(mle_null[0])
        mle_val = likelihood(mle)
        chi2 = 1.0 - stats.chi2.cdf(2.0 * (null_val - mle_val), 1)
        if verbose:
            logging.info(
                f"Likelihood comparison: chi2={chi2:.3f}, "
                f"mle_rho={mle[0]:.3f}, mle_sigma={mle[1]:.3f}, "
                f"mle_null_sigma={mle_null[0]:.3f}"
            )
    except Exception as err:
        if verbose:
            logging.warning(f"Chi-squared computation failed: {err}")
        return True

    return chi2 > 0.05


def gram_schmidt(matrix):
    """Gram-Schmidt orthonormalization of matrix rows.

    Parameters
    ----------
    matrix : array-like, shape=[n_vectors, dim]
        Input matrix rows to orthonormalize.

    Returns
    -------
    orthonormal_matrix : array-like, shape=[n_orthonormal, dim]
        Orthonormalized rows where n_orthonormal <= n_vectors.
    """
    matrix = gs.array(matrix)
    n_vectors, dim = matrix.shape
    orthonormal_vectors = []

    for i in range(n_vectors):
        vector = matrix[i]

        # Subtract projections onto previous orthonormal vectors
        for ortho_vector in orthonormal_vectors:
            vector = vector - gs.sum(vector * ortho_vector) * ortho_vector

        # Check if vector is linearly independent
        norm = gs.linalg.norm(vector)
        if norm > gs.atol:
            vector = vector / norm
            orthonormal_vectors.append(vector)

    if orthonormal_vectors:
        return gs.stack(orthonormal_vectors)
    else:
        return gs.empty((0, dim))


class PrincipalNestedSpheres(BaseEstimator, TransformerMixin):
    """Principal Nested Spheres for recursive dimension reduction on spheres.

    Parameters
    ----------
    space : Hypersphere
        Ambient sphere geometry object.
    n_init : int
        Number of random initializations for robust fitting.
        Optional, default: 10.
    max_iter : int
        Maximum number of optimization iterations per subsphere fit.
        Optional, default: 1000.
    tol : float
        Convergence tolerance for optimization.
        Optional, default: 1e-8.
    sphere_mode : str, {"adaptive", "great", "small"}
        Sphere fitting mode.
        Optional, default: "adaptive".
    verbose : bool
        If True, print debug information during fitting.
        Optional, default: False.

    Attributes
    ----------
    nested_spheres_ : list of tuple
        Sequence of (normal, height) defining each fitted subsphere.
    residuals_ : list of array-like
        Residual distances for each nested subsphere fit.
    mean_ : array-like, shape=[2]
        Final nested mean on S^1 after reduction.

    Notes
    -----
    The algorithm recursively fits codimension-1 subspheres to project data
    onto successively lower-dimensional spheres until reaching S^1.

    References
    ----------
    Original implementation: https://github.com/dauletbeck/RNA-Classification
    """

    def __init__(
        self,
        space,
        n_init=10,
        max_iter=1000,
        tol=1e-8,
        sphere_mode="adaptive",
        verbose=False,
    ):
        self.space = space
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.sphere_mode = sphere_mode
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit a sequence of nested subspheres to data.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Points on the ambient sphere.
        y : None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = gs.array(X)
        X = X / gs.linalg.norm(X, axis=1, keepdims=True)
        self.nested_spheres_ = []
        self.residuals_ = []

        # Ambient dimension k: sphere S^k subset of R^{k+1}
        current_dim = X.shape[1] - 1
        current_data = X

        # Recursively fit codimension-1 subspheres until dimension == 1
        while current_dim >= 1:
            normal, height = self._fit_subsphere(current_data)
            self.nested_spheres_.append((normal, height))

            if self.verbose:
                logging.info(
                    f"Fitted S^{current_dim} in R^{current_dim + 1}: "
                    f"normal={normal}, height={height:.6f}"
                )

            # Compute residuals for this subsphere
            residuals = self._compute_signed_distances(current_data, normal, height)
            self.residuals_.append(residuals)

            # Project onto the subsphere and drop one dimension
            current_data = self._project_to_subsphere(
                current_data, normal, height, return_reduced=True
            )
            current_dim -= 1

            if current_data.shape[1] == 2:
                break

        # Compute final circular mean on S^1
        self.mean_ = self._circular_mean(current_data)
        return self

    def transform(self, X):
        """Project data through the fitted nested subspheres.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_original_features]
            Input points on the original sphere.

        Returns
        -------
        X_transformed : array-like, shape=[n_samples, 2]
            Points on the final S^1 after nested projections.
        """
        X_projected = gs.array(X)
        for normal, height in self.nested_spheres_:
            X_projected = self._project_to_subsphere(
                X_projected, normal, height, return_reduced=True
            )
        return X_projected

    def fit_transform(self, X, y=None):
        """Fit the estimator and apply nested projections in one step.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Input points on the sphere.
        y : None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        X_transformed : array-like, shape=[n_samples, 2]
            Reduced representation on S^1.
        """
        return self.fit(X).transform(X)

    def _fit_subsphere(self, points):
        """Fit the best codimension-1 subsphere using the specified mode.

        Parameters
        ----------
        points : array-like, shape=[n_samples, n_features]
            Points on the current sphere.

        Returns
        -------
        normal : array-like, shape=[n_features]
            Unit normal vector defining the subsphere center direction.
        height : float
            Height (cosine of radius) of the subsphere center from origin.
        """
        if self.sphere_mode == "adaptive":
            return self._fit_adaptive_sphere(points)
        elif self.sphere_mode == "great":
            return self._fit_great_sphere(points)
        elif self.sphere_mode == "small":
            return self._fit_small_sphere(points)
        else:
            raise ValueError(f"Unknown sphere_mode: {self.sphere_mode}")

    def _fit_adaptive_sphere(self, points):
        """Adaptively choose between great and small sphere fitting."""
        small_fit = None
        try:
            small_fit = self._fit_small_sphere(points)
        except Exception:
            small_fit = None

        if small_fit is None:
            return self._fit_great_sphere(points)

        small_normal, small_height = small_fit
        if gs.abs(small_height) < gs.atol:
            return small_normal, small_height

        radii = self._projected_radii(points, small_normal)
        prefer_great = _compare_likelihoods(
            radii,
            points.shape[1] - 1,
            verbose=self.verbose,
        )

        if prefer_great:
            try:
                return self._fit_great_sphere(points)
            except Exception:
                return small_normal, small_height

        return small_normal, small_height

    def _fit_great_sphere(self, points):
        """Fit a great sphere using multiple initializations."""
        best_normal = None
        best_score = float("inf")
        old_directions = []

        for _ in range(self.n_init):
            try:
                normal = self._fit_single_great_sphere(points, old_directions)
                score = self._calculate_sphere_score(points, normal, 0.0)

                if score < best_score:
                    best_score = score
                    best_normal = normal

                old_directions.append(normal)

            except Exception:
                continue

        if best_normal is None:
            # Fallback to mean direction
            mean_direction = gs.mean(points, axis=0)
            best_normal = mean_direction / gs.linalg.norm(mean_direction)

        return best_normal, 0.0

    def _fit_single_great_sphere(self, points, old_directions=None):
        """Fit a single great sphere using least squares optimization."""
        # Generate initial direction
        normal = self._generate_seed(points.shape[1], old_directions)

        def objective(direction):
            return self._great_sphere_objective(direction, points)

        result = leastsq(
            objective, normal, xtol=self.tol, ftol=self.tol, maxfev=self.max_iter
        )

        if len(result) < 2:
            raise RuntimeError("Great sphere optimization failed")

        fitted_normal = result[0]
        fitted_normal = fitted_normal / gs.linalg.norm(fitted_normal)
        return fitted_normal

    def _great_sphere_objective(self, direction, points):
        """Objective function for great sphere fitting."""
        norm_direction = gs.linalg.norm(direction)
        if norm_direction < gs.atol:
            return gs.ones(len(points)) * 1e6

        normal = direction / norm_direction
        return self._compute_signed_distances(points, normal, 0.0)

    def _fit_small_sphere(self, points):
        """Fit a small sphere with robust multiple initialization."""
        best_normal = None
        best_height = 0.0
        best_score = float("inf")
        old_directions = []

        for _ in range(self.n_init):
            try:
                normal, height = self._fit_single_small_sphere(points, old_directions)
                score = self._calculate_sphere_score(points, normal, height)

                if score < best_score:
                    best_score = score
                    best_normal = normal
                    best_height = height

                old_directions.append(normal)

            except Exception:
                continue

        if best_normal is None:
            # Fallback to great sphere
            return self._fit_great_sphere(points)

        return best_normal, best_height

    def _fit_single_small_sphere(self, points, old_directions=None):
        """Fit a single small sphere using the RNA repository approach."""
        # Generate initial direction
        normal = self._generate_seed(points.shape[1], old_directions)

        result = leastsq(
            self._small_sphere_objective,
            normal,
            args=(points,),
            xtol=self.tol,
            ftol=self.tol,
            maxfev=self.max_iter,
        )

        if len(result) < 2:
            raise RuntimeError("Small sphere optimization failed")

        fitted_normal = result[0]
        fitted_normal = fitted_normal / gs.linalg.norm(fitted_normal)

        # Calculate height from projections
        projections = gs.matvec(points, fitted_normal)
        height = gs.mean(projections)

        return fitted_normal, height

    def _generate_seed(self, dim, old_directions=None):
        """Generate random unit vector with angular separation from existing directions.

        Based on the RNA repository implementation.
        """
        if old_directions is None or len(old_directions) == 0:
            normal = 2 * gs.random.uniform(size=(dim,)) - 1
            return normal / gs.linalg.norm(normal)

        out = old_directions[0].copy()
        max_attempts = 100
        attempts = 0

        # Ensure angular separation (cosine < 0.7)
        while attempts < max_attempts:
            old_array = gs.stack(old_directions)
            cosines = gs.abs(gs.matvec(old_array, out))
            if not gs.any(cosines > 0.7):
                break

            out = 2 * gs.random.uniform(size=(dim,)) - 1
            norm = gs.linalg.norm(out)
            if norm < gs.atol:
                out = gs.ones(dim) / gs.sqrt(dim)
            else:
                out = out / norm
            attempts += 1

        return out

    def _small_sphere_objective(self, direction, points):
        """Objective function with norm constraint from RNA repository.

        Direction with norm constraint.
        """
        norm_direction = gs.linalg.norm(direction)
        if norm_direction < gs.atol:
            angles = gs.zeros(points.shape[0])
        else:
            normalized_direction = direction / norm_direction
            dot_products = gs.matvec(points, normalized_direction)
            angles = gs.arcsin(gs.clip(dot_products, -1, 1))

        residuals = angles - gs.mean(angles)
        norm_constraint = norm_direction - 1
        return gs.concatenate([residuals, [norm_constraint]])

    def _compute_signed_distances(self, points, normal, height):
        """Calculate signed distances from points to the sphere."""
        # Calculate angular distances using geomstats sphere distance
        cos_angles = gs.matvec(points, normal)
        angles = gs.arccos(gs.clip(cos_angles, -1, 1))

        # For great sphere (height=0), target angle is π/2
        if gs.abs(height) < gs.atol:
            return angles - gs.pi / 2
        else:
            # For small sphere, calculate distance from sphere radius
            sphere_radius = gs.arccos(gs.clip(gs.abs(height), 0, 1))
            return angles - sphere_radius

    def _calculate_sphere_score(self, points, normal, height):
        """Calculate score for sphere fit quality."""
        residuals = self._compute_signed_distances(points, normal, height)
        return gs.sum(residuals**2)

    def _projected_radii(self, points, normal):
        """Compute angular radii of points with respect to a direction."""
        projections = gs.matvec(points, normal)
        projections = gs.clip(gs.abs(projections), 0.0, 1.0)
        return gs.arccos(projections)

    def _project_to_subsphere(self, points, normal, height, return_reduced=True):
        """Project points onto a fitted subsphere and optionally reduce dimension.

        Parameters
        ----------
        points : array-like, shape=[n_samples, n_features]
            Points on the ambient sphere.
        normal : array-like, shape=[n_features]
            Unit normal vector of the subsphere's center.
        height : float
            Height (cosine of radius) of the subsphere center.
        return_reduced : bool
            If True, return intrinsic coordinates dropping one dimension.
            Optional, default: True.

        Returns
        -------
        projected_points : array-like
            Projected points.
        """
        projection_center = height * normal
        flat_radius = gs.sqrt(gs.maximum(0, 1.0 - height**2))

        points_minus = points - gs.outer(gs.matvec(points, normal), normal)
        norms = gs.linalg.norm(points_minus, axis=1, keepdims=True)
        norms = gs.where(norms < gs.atol, 1.0, norms)  # Avoid division by zero

        projected_points = projection_center + flat_radius * (points_minus / norms)

        if not return_reduced:
            return projected_points

        # Intrinsic drop-one-dimension via Householder reflection
        dim = normal.shape[0]
        householder_vector = normal.copy()
        sign = 1.0 if normal[0] >= 0 else -1.0
        householder_vector = gs.array(householder_vector)
        householder_vector = gs.assignment(
            householder_vector, sign * gs.linalg.norm(normal) + normal[0], 0
        )
        vector_norm = gs.linalg.norm(householder_vector)

        if vector_norm < gs.atol:
            # Handle degenerate case
            householder_matrix = gs.eye(dim)
        else:
            householder_vector = householder_vector / vector_norm
            householder_matrix = gs.eye(dim) - 2.0 * gs.outer(
                householder_vector, householder_vector
            )

        orthogonal_basis = householder_matrix[:, 1:]

        reduced_points = projected_points - projection_center
        if flat_radius > gs.atol:
            reduced_points = reduced_points / flat_radius

        reduced_points = gs.matmul(reduced_points, orthogonal_basis)
        return reduced_points

    def _circular_mean(self, points):
        """Compute the circular mean on S^1.

        Parameters
        ----------
        points : array-like, shape=[n_samples, 2]
            Points on S^1.

        Returns
        -------
        mean : array-like, shape=[2]
            Unit vector representing the mean direction on S^1.
        """
        angles = gs.arctan2(points[:, 1], points[:, 0])
        mean_angle = gs.arctan2(gs.mean(gs.sin(angles)), gs.mean(gs.cos(angles)))
        return gs.array([gs.cos(mean_angle), gs.sin(mean_angle)])
