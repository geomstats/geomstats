import geomstats.backend as gs

from ._sklearn import (
    EuclideanInputMixin,
    EuclideanInputOutputMixin,
    _enable_array_dispatch,
)
from ._sklearngs.linear_model._base import LinearRegression as _LinearRegression

_enable_array_dispatch()


class VectorValuedLinearRegression(EuclideanInputMixin, _LinearRegression):
    """Linear regression with structured Euclidean inputs and vector-valued targets.

    This estimator extends sklearn's linear regression to inputs represented as
    points in a Euclidean space with nontrivial shape. Inputs are flattened before
    fitting and prediction, while fitted coefficients are reshaped back to the
    shape of the input space.

    The target values follow sklearn's usual convention: scalar targets have
    shape ``(n_samples,)`` and vector-valued or multi-output targets have shape
    ``(n_samples, n_outputs)``.

    Parameters
    ----------
    space : Euclidean
        Euclidean input space. Its ``shape`` attribute determines the structured
        shape of each input point.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        Whether to copy the input array before fitting.
    tol : float, default=1e-6
        Precision of the solution.
    n_jobs : int or None, default=None
        Number of jobs to use for the computation.
    positive : bool, default=False
        Whether to force the coefficients to be positive.

    Attributes
    ----------
    coef_ : array-like, shape=(n_features,) or (n_targets, n_features)
        Estimated linear coefficients in sklearn's flattened feature
        representation.
    coef_reshaped_ : array-like, shape=space.shape or (n_targets, *space.shape)
        Estimated linear coefficients reshaped to the structured input space. For
        scalar-valued targets, ``coef_reshaped_`` has shape ``space.shape``. For
        vector-valued targets, its first axis indexes the target component and
        the remaining axes match ``space.shape``.
    intercept_ : float or array-like, shape=(n_targets,)
        Independent term in the linear model. It is a scalar for scalar-valued
        targets and a vector for vector-valued targets.
    """

    def __init__(
        self,
        space,
        *,
        fit_intercept=True,
        copy_X=True,
        tol=1e-6,
        n_jobs=None,
        positive=False,
    ):
        self.space = space
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            tol=tol,
            n_jobs=n_jobs,
            positive=positive,
        )

    def _reshape_fitted_attrs(self):
        coef = gs.reshape(
            self.coef_,
            (*self.coef_.shape[:-1], *self.space.shape),
        )
        self._set_reshaped_attr("coef_", coef)


class TensorValuedLinearRegression(EuclideanInputOutputMixin, _LinearRegression):
    """Linear regression with structured Euclidean inputs and tensor-valued outputs.

    This estimator extends sklearn's linear regression to inputs and outputs
    represented as points in Euclidean spaces with nontrivial shapes. Inputs and
    outputs are flattened before fitting.

    Parameters
    ----------
    space : Euclidean
        Euclidean input space. Its ``shape`` attribute determines the structured
        shape of each input point.
    image_space : Euclidean
        Euclidean output space. Its ``shape`` attribute determines the structured
        shape of each output point.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        Whether to copy the input array before fitting.
    tol : float, default=1e-6
        Precision of the solution.
    n_jobs : int or None, default=None
        Number of jobs to use for the computation.
    positive : bool, default=False
        Whether to force the coefficients to be positive.

    Attributes
    ----------
    coef_ : array-like, shape=(prod(image_space.shape), prod(space.shape))
        Estimated linear coefficients in sklearn's flattened representation.
        The first axis indexes flattened output coordinates and the second axis
        indexes flattened input coordinates.
    coef_reshaped_ : array-like, shape=(*image_space.shape, *space.shape)
        Estimated linear coefficients reshaped as a linear map from structured
        inputs in ``space`` to structured outputs in ``image_space``. The leading
        axes match ``image_space.shape`` and the trailing axes match
        ``space.shape``.
    intercept_ : array-like, shape=(prod(image_space.shape),)
        Independent term in sklearn's flattened output representation.
    intercept_reshaped_ : array-like, shape=image_space.shape
        Independent term reshaped as a point in the output space.
    """

    def __init__(
        self,
        space,
        image_space,
        *,
        fit_intercept=True,
        copy_X=True,
        tol=1e-6,
        n_jobs=None,
        positive=False,
    ):
        self.space = space
        self.image_space = image_space

        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            tol=tol,
            n_jobs=n_jobs,
            positive=positive,
        )

    def _reshape_fitted_attrs(self):
        coef = gs.reshape(
            self.coef_,
            (*self.image_space.shape, *self.space.shape),
        )
        self._set_reshaped_attr("coef_", coef)

        if self.fit_intercept:
            intercept = gs.reshape(self.intercept_, self.image_space.shape)
            self._set_reshaped_attr("intercept_", intercept)

    def _reshape_fitted_attrs(self):
        coef = gs.reshape(self.coef_, (*self.image_space.shape, *self.space.shape))
        intercept = gs.reshape(self.intercept_, self.image_space.shape)

        self._set_reshaped_attr("coef_", coef)
        self._set_reshaped_attr("intercept_", intercept)

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        Matrix-valued outputs are flattened per sample before calling sklearn's
        ``r2_score``.
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)

        y = gs.reshape(y, (len(y), -1))
        y_pred = gs.reshape(y_pred, (len(y_pred), -1))

        return r2_score(y, y_pred, sample_weight=sample_weight)


def LinearRegression(space, image_space=None, **kwargs):
    """Create a linear regression estimator for structured Euclidean data.

    This factory returns a linear regression estimator adapted to the geometry of
    the input and output spaces.

    If ``image_space`` is ``None``, the returned estimator accepts structured
    Euclidean inputs and scalar- or vector-valued targets following sklearn's
    standard target conventions.

    If ``image_space`` is provided, the returned estimator accepts structured
    Euclidean inputs and predicts structured outputs in ``image_space``.

    Parameters
    ----------
    space : Euclidean
        Euclidean input space.
    image_space : Euclidean or None, default=None
        Euclidean output space. If ``None``, a
        ``VectorValuedLinearRegression`` is returned. Otherwise, a
        ``MatrixValuedLinearRegression`` is returned.
    **kwargs : dict
        Additional keyword arguments passed to the selected estimator.

    Returns
    -------
    estimator : VectorValuedLinearRegression or MatrixValuedLinearRegression
        Linear regression estimator adapted to the provided spaces.
    """
    if image_space is None or len(image_space.shape) < 2:
        return VectorValuedLinearRegression(space, **kwargs)

    return TensorValuedLinearRegression(space, image_space, **kwargs)
