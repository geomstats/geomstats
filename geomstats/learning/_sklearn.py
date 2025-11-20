"""Collections of wrappers around scikit-learn.

Main goal is to make them compatible with geomstats.
Common issues are point shapes (in geomstats, points may
be represented as n-dim arrays)
and backend support.

For maintainability reasons, we wrap a sklearn object only
when it is strictly necessary.
"""

import os

from sklearn.base import RegressorMixin as _RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA as _PCA
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import geomstats.backend as gs

SCIPY_ARRAY_API = True if os.environ.get("SCIPY_ARRAY_API", False) == "1" else False


class RegressorMixin(_RegressorMixin):
    """Mixin class for all regression estimators in scikit-learn."""

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction."""
        y_pred = self.predict(X)
        return r2_score(
            gs.to_numpy(y), gs.to_numpy(y_pred), sample_weight=sample_weight
        )


class GetParamsMixin:
    # to avoid pprint error when not storing

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                continue

            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


class InvertibleFlattenButFirst(FunctionTransformer):
    """Data reshaper.

    Reshapes data and transforms it into proper backend when inverting.
    """

    def __init__(self):
        super().__init__(
            func=self._optional_flatten,
            inverse_func=self._optional_reshape,
            check_inverse=False,
        )
        self.shape = None

    @staticmethod
    def _optional_flatten(array):
        """Optionally flatten array.

        Flattens array if ndim != 2.
        """
        if array.ndim == 2:
            return array
        return array.reshape((array.shape[0], -1))

    def _optional_reshape(self, array):
        """Optionally reshape array.

        Reshapes array if ndim != 2.
        Additionally, converts back to proper tensor type
        if `SCIPY_ARRAY_API` is not activated.
        """
        out = array if self.shape is None else array.reshape(self.shape)
        if SCIPY_ARRAY_API:
            return out

        return gs.from_numpy(out)

    def fit(self, X, y=None):
        """Fit transform.

        Parameters
        ----------
        X : {array-like, sparse-matrix} of shape (n_samples, n_features) \
                if `validate=True` else any object that `func` can handle
            Input array.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        if X.ndim == 2:
            return self
        self.shape = (-1,) + X.shape[1:]
        return self


class ModelAdapter(Pipeline):
    """Adapter sklearn model.

    Handles input/output transformations in a pipeline-way.
    In particular, reshapes points and transforms them into
    appropriate backend.

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Estimator to be adapted.
    """

    def __init__(self, model):
        self.model = model
        self._regression = False

        if isinstance(model, _RegressorMixin):
            self._regression = True
            model = TransformedTargetRegressor(
                regressor=model,
                transformer=InvertibleFlattenButFirst(),
                check_inverse=False,
            )

        steps = [
            ("reshape", InvertibleFlattenButFirst()),
            ("estimator", model),
        ]

        super().__init__(steps=steps)

    def get_model(self):
        """Get fitted model."""
        if self._regression:
            return self.named_steps["estimator"].regressor_

        return self.named_steps["estimator"]

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to fitted model.
        If name starts with ``reshaped``, then inverse transform
        is applied to recover expected shape/backend.
        """
        model_ = self.get_model()

        if name.startswith("reshaped_") and name.endswith("_"):
            out = getattr(model_, name[9:])
            out_ = self.named_steps["reshape"].inverse_transform(out)
            if out.ndim == 1:
                return out_[0]

            return out_

        out = getattr(model_, name)
        if name.endswith("_") and not SCIPY_ARRAY_API:
            return gs.from_numpy(out)

        return out


class LinearRegression(GetParamsMixin, ModelAdapter):
    """Ordinary least squares Linear Regression.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    for details.
    """

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        regressor = _LinearRegression(
            fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive
        )
        super().__init__(regressor)


class PCA(GetParamsMixin, ModelAdapter):
    """Principal component analysis (PCA).

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    for details.
    """

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        pca = _PCA(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )
        super().__init__(pca)
