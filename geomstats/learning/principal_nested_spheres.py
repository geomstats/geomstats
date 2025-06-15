import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from scipy.optimize import least_squares

class PrincipalNestedSpheres:
    """
    Principal Nested Spheres (PNS) for recursive dimension reduction on spheres.

    Parameters
    ----------
    sphere : Hypersphere
        The ambient sphere geometry object (defines dimension and operations).
    max_iter : int
        Maximum number of optimization iterations per fit.
    tol : float
        Convergence tolerance.
    verbose : bool
        Verbose output for debugging.
    """

    def __init__(self, sphere, max_iter=50, tol=1e-8, verbose=False):
        self.sphere = sphere
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        """
        Fit a sequence of nested spheres to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input points on the sphere.
        """
        self.nested_spheres_ = []
        self.residuals_ = []
        X_rec = np.array(X)
        n = X_rec.shape[1] - 1

        while n >= 1:
            # Fit subsphere (codim-1 sphere) to X_rec
            normal, height = self._fit_subsphere(X_rec)
            self.nested_spheres_.append((normal, height))

            if self.verbose:
                print(f"Fitted S^{n} in R^{n+1}: normal={normal}, height={height:.6f}")

            # Project points onto subsphere (lower-dim sphere)
            X_rec = self._project_to_subsphere(X_rec, normal, height)
            n -= 1

            # Optional: store residuals (distances to fitted sphere)
            dist = np.arccos(np.clip(np.dot(X_rec, normal), -1, 1)) - np.arccos(height)
            self.residuals_.append(dist)

            # If on the circle (2D), break after this
            if X_rec.shape[1] == 2:
                break

        self._final_mean = self._circular_mean(X_rec)
        return self

    def transform(self, X):
        """
        Transform data X to the 1D representation via the sequence of projections.
        Returns the representation on the final S^1.
        """
        X_proj = np.array(X)
        for (normal, height) in self.nested_spheres_:
            X_proj = self._project_to_subsphere(X_proj, normal, height)
        return X_proj

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # --- Core geometric steps ---

    def _fit_subsphere(self, X):
        """
        Fit the best subsphere to the data in the least squares sense.
        Returns (normal, height).
        """
        # Use the mean direction as initialization
        mean = np.mean(X, axis=0)
        normal = mean / np.linalg.norm(mean)
        # Optimize over normal and height
        def residuals(params):
            normal = params[:-1]
            normal /= np.linalg.norm(normal)
            height = params[-1]
            proj = height * normal
            # Project each x onto the subsphere center, then get distance
            cos_angles = np.dot(X, normal)
            angles = np.arccos(np.clip(cos_angles, -1, 1))
            sphere_radius = np.arccos(np.clip(height, -1, 1))
            return angles - sphere_radius

        # Initial guess: normal is mean dir, height is 0 (great sphere)
        x0 = np.hstack([normal, 0.0])
        result = least_squares(residuals, x0, xtol=self.tol, ftol=self.tol, max_nfev=self.max_iter)
        fit_normal = result.x[:-1]
        fit_normal /= np.linalg.norm(fit_normal)
        fit_height = np.clip(result.x[-1], -1, 1)
        return fit_normal, fit_height

    def _project_to_subsphere(self, X, normal, height):
        """
        Project points X onto the fitted subsphere defined by (normal, height).
        Returns projected points (lower-dimensional sphere).
        """
        proj_center = height * normal
        flat_radius = np.sqrt(1 - height ** 2)
        # For each x, subtract component along normal, then normalize and scale
        X_minus = X - np.outer(np.dot(X, normal), normal)
        X_proj = proj_center + flat_radius * X_minus / np.linalg.norm(X_minus, axis=1, keepdims=True)
        return X_proj

    def _circular_mean(self, X):
        """
        Compute the circular mean (for the final S^1 step).
        """
        angle = np.arctan2(X[:, 1], X[:, 0])
        mean_angle = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
        return np.array([np.cos(mean_angle), np.sin(mean_angle)])

    @property
    def mean_(self):
        """Return the final nested mean (on S^1)."""
        return self._final_mean

