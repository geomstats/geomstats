r"""
Principal Nested Spheres (PNS).

Defines the PrincipalNestedSpheres class for recursive dimension
reduction on spheres as described in: TODO: add reference
"""

import numpy as np
from scipy.optimize import least_squares

from geomstats.geometry.hypersphere import Hypersphere


class PrincipalNestedSpheres:
    r"""
    Principal Nested Spheres (PNS) for recursive dimension reduction on spheres.

    Implements the algorithm as described in: TODO: add full reference.

    Parameters
    ----------
    sphere : Hypersphere
        Ambient sphere geometry object (defines dimension and operations).
    max_iter : int, optional
        Maximum number of optimization iterations per subsphere fit. Default is 50.
    tol : float, optional
        Convergence tolerance for least squares. Default is 1e-8.
    verbose : bool, optional
        If True, print debug information during fitting. Default is False.

    Attributes
    ----------
    nested_spheres_ : list of tuple
        Sequence of (normal, height) defining each fitted subsphere.
    residuals_ : list of array-like
        Residual distances for each nested subsphere fit.
    mean_ : array-like, shape=(2,)
        Final nested mean on S^1 after reduction.
    """

    def __init__(
        self,
        sphere: Hypersphere,
        max_iter: int = 50,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.sphere = sphere
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: np.ndarray) -> "PrincipalNestedSpheres":
        """
        Fit a sequence of nested subspheres to data.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Points on the ambient sphere.

        Returns
        -------
        self : PrincipalNestedSpheres
            Fitted estimator.
        """
        X_rec = np.asarray(X)
        self.nested_spheres_ = []
        self.residuals_ = []

        # Ambient dimension k: sphere S^k subset of R^{k+1}
        n = X_rec.shape[1] - 1

        # Recursively fit codimension-1 subspheres until dimension == 1
        while n >= 1:
            normal, height = self._fit_subsphere(X_rec)
            self.nested_spheres_.append((normal, height))

            if self.verbose:
                print(
                    f"Fitted S^{n} in R^{n + 1}: normal={normal}, height={height:.6f}"
                )

            # Compute residuals for this subsphere
            cos_angles = X_rec.dot(normal)
            sphere_radius = np.arccos(height)
            dist = np.arccos(np.clip(cos_angles, -1.0, 1.0)) - sphere_radius
            self.residuals_.append(dist)

            # Project onto the subsphere and drop one dimension
            X_rec = self._project_to_subsphere(
                X_rec, normal, height, return_reduced=True
            )
            n -= 1

            if X_rec.shape[1] == 2:
                break

        # Compute final circular mean on S^1
        self._final_mean = self._circular_mean(X_rec)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data through the fitted nested subspheres.

        Each step projects onto a subsphere and reduces dimension by one.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_original)
            Input points on the original sphere.

        Returns
        -------
        X_out : array-like, shape=(n_samples, 2)
            Points on the final S^1 after nested projections.
        """
        X_proj = np.asarray(X)
        for normal, height in self.nested_spheres_:
            X_proj = self._project_to_subsphere(X_proj, normal, height)
        return X_proj

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the estimator and apply nested projections in one step.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input points on the sphere.

        Returns
        -------
        X_out : array-like, shape=(n_samples, 2)
            Reduced representation on S^1.
        """
        return self.fit(X).transform(X)

    def _fit_subsphere(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit the best codimension-1 subsphere by least squares.

        Parameters
        ----------
        X : array-like, shape=(n_samples, k+1)
            Points on S^k.

        Returns
        -------
        normal : array-like, shape=(k+1,)
            Unit normal vector defining the subsphere center direction.
        height : float
            Height (cosine of radius) of the subsphere center from origin.
        """
        # Initialization: mean direction
        mean_dir = X.mean(axis=0)
        normal = mean_dir / np.linalg.norm(mean_dir)

        def residuals(params):
            n = params[:-1]
            n = n / np.linalg.norm(n)
            h = params[-1]
            angles = np.arccos(np.clip(X.dot(n), -1.0, 1.0))
            sphere_radius = np.arccos(np.clip(h, -1.0, 1.0))
            return angles - sphere_radius

        x0 = np.hstack([normal, 0.0])
        result = least_squares(
            residuals, x0, xtol=self.tol, ftol=self.tol, max_nfev=self.max_iter
        )

        fit_normal = result.x[:-1]
        fit_normal /= np.linalg.norm(fit_normal)
        fit_height = float(np.clip(result.x[-1], -1.0, 1.0))
        return fit_normal, fit_height

    def _project_to_subsphere(
        self,
        X: np.ndarray,
        normal: np.ndarray,
        height: float,
        return_reduced: bool = True,
    ) -> np.ndarray:
        """
        Project points onto a fitted subsphere and optionally reparameterize.

        Parameters
        ----------
        X : array-like, shape=(n_samples, k+1)
            Points on the ambient sphere S^k.
        normal : array-like, shape=(k+1,)
            Unit normal vector of the subsphere's center.
        height : float
            Height (cosine of radius) of the subsphere center.
        return_reduced : bool, optional
            If True, return intrinsic coordinates in R^k (dropping one dim).
            If False, return extrinsic projection in R^{k+1}.
            Default is True.

        Returns
        -------
        X_out : array-like
            Projected points of shape
            - (n_samples, k) if return_reduced is True,
            - (n_samples, k+1) otherwise.
        """
        proj_center = height * normal
        flat_radius = np.sqrt(1.0 - height**2)

        X_minus = X - np.outer(X.dot(normal), normal)
        X_proj = proj_center + flat_radius * (
            X_minus / np.linalg.norm(X_minus, axis=1, keepdims=True)
        )

        if not return_reduced:
            return X_proj

        # Intrinsic drop-one-dimension via Householder
        p = normal.shape[0]
        v = normal.copy()
        v[0] += np.copysign(np.linalg.norm(normal), normal[0])
        v /= np.linalg.norm(v)
        H = np.eye(p) - 2.0 * np.outer(v, v)
        U = H[:, 1:]

        Z_unit = (X_proj - proj_center) / flat_radius
        X_red = Z_unit @ U
        return X_red

    def _circular_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the circular mean on S^1.

        Parameters
        ----------
        X : array-like, shape=(n_samples, 2)
            Points on S^1.

        Returns
        -------
        mean : array-like, shape=(2,)
            Unit vector representing the mean direction on S^1.
        """
        angles = np.arctan2(X[:, 1], X[:, 0])
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        return np.array([np.cos(mean_angle), np.sin(mean_angle)])

    @property
    def mean_(self) -> np.ndarray:
        """
        Return the final nested mean on S^1.

        Returns
        -------
        mean : array-like, shape=(2,)
            Nested mean after full reduction.
        """
        return self._final_mean
