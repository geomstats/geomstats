"""Collections of wrappers around scikit-learn.

Main goal is to make them compatible with geomstats.
Common issues are point shapes (in geomstats, points may
be represented as n-dim arrays)
and backend support.

For maintainability reasons, we wrap a sklearn object only
when it is strictly necessary.
"""

import os
from contextlib import contextmanager
from functools import wraps

import numpy as np
from sklearn import set_config
from sklearn.base import RegressorMixin as _RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data

import geomstats.backend as gs

SCIPY_ARRAY_API = True if os.environ.get("SCIPY_ARRAY_API", False) == "1" else False


def _enable_array_dispatch():
    if gs.__name__.endswith("pytorch"):
        set_config(array_api_dispatch=True)


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


@contextmanager
def temporary_attrs(objs, names, values):
    old_values = []
    for obj, name, value in zip(objs, names, values):
        old_values.append(getattr(obj, name))
        setattr(obj, name, value)

    try:
        yield
    finally:
        for obj, name, old_value in zip(objs, names, old_values):
            setattr(obj, name, old_value)


def validate_data_skip_check_array(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    reset=True,
    validate_separately=False,
    skip_check_array=False,
    **check_params,
):
    return validate_data(
        _estimator,
        X=X,
        y=y,
        reset=reset,
        validate_separately=validate_separately,
        skip_check_array=True,
        **check_params,
    )


def check_array_allow_nd(
    array,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_writeable=False,
    ensure_all_finite=True,
    ensure_non_negative=False,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):
    return check_array(
        array,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_writeable=force_writeable,
        ensure_all_finite=ensure_all_finite,
        ensure_non_negative=ensure_non_negative,
        ensure_2d=ensure_2d,
        allow_nd=True,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name=input_name,
    )


class SklearnInteropMixin:
    """Mixin to patch sklearn validation and backend-sensitive routines.

    This mixin is intended for sklearn-compatible estimators that operate on
    data not natively supported by sklearn validation utilities, such as
    manifold-valued objects, higher-dimensional array representations, or
    backend tensors requiring custom dispatch.

    Subclasses configure the behavior by setting:

    - `_sklearn_patches`: triples of `(module, name, value)` describing temporary
    attribute replacements to apply.
    - `_patched_methods`: public method names that should run under the patch
    context.
    - `_use_sklearn_patches`: whether the patches should be active for this
    instance.

    The patches are applied only while one of the patched methods is executing.
    """

    _sklearn_patches = ()
    _patched_methods = ()
    _use_sklearn_patches = False

    @contextmanager
    def _patch_context(self):
        """Temporarily install object-aware validation helpers.

        The configured module attributes are restored when the context exits,
        including when the wrapped method raises an exception.
        """
        patches = super().__getattribute__("_sklearn_patches")
        modules, names, values = zip(*patches)

        with temporary_attrs(modules, names, values):
            yield

    def __getattribute__(self, name):
        """Wrap selected public methods in the object-validation context."""
        attr = super().__getattribute__(name)

        if name.startswith("_") or not self._use_sklearn_patches:
            return attr

        wrapped_names = super().__getattribute__("_patched_methods")
        if name not in wrapped_names or not callable(attr):
            return attr

        @wraps(attr)
        def wrapped(*args, **kwargs):
            with self._patch_context():
                return attr(*args, **kwargs)

        return wrapped


class OutputToBackendMixin:
    """Mixin converting selected NumPy outputs to the active Geomstats backend."""

    _output_to_backend_methods = ()

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name.startswith("_") or gs.__name__.endswith("numpy"):
            return attr

        methods = super().__getattribute__("_output_to_backend_methods")
        if name not in methods or not callable(attr):
            return attr

        @wraps(attr)
        def wrapped(*args, **kwargs):
            return self._output_to_backend(attr(*args, **kwargs))

        return wrapped

    def _output_to_backend(self, results):
        def convert(result):
            if isinstance(result, np.ndarray):
                return gs.from_numpy(result)
            return result

        if isinstance(results, tuple):
            return tuple(convert(item) for item in results)

        return convert(results)


class EuclideanInputMixin:
    """Mixin flattening Euclidean structured inputs for sklearn estimators.

    Public methods accept inputs with shape ``(n_samples, *space.shape)``.
    The wrapped sklearn estimator receives inputs with shape
    ``(n_samples, prod(space.shape))``.

    Subclasses may override ``_reshape_attrs`` to expose fitted attributes
    reshaped back to ``space.shape``.
    """

    _reshape_input_methods = ("fit", "predict", "score")
    _reshaped_attr_suffix = "_reshaped_"

    def _flatten_X(self, X):
        """Flatten structured inputs to sklearn's 2D convention."""
        point_shape = self.space.shape

        if X.shape[1:] != point_shape:
            raise ValueError(
                f"Expected X with shape (n_samples, {point_shape}), " f"got {X.shape}."
            )

        return X.reshape((X.shape[0], -1))

    def _reshape_attrs(self):
        """Expose reshaped aliases for fitted sklearn attributes.

        Subclasses can override this method. The default does nothing.
        """
        return None

    def _set_reshaped_attr(self, name, value):
        """Set a reshaped alias for a fitted sklearn attribute.

        Examples
        --------
        ``name="coef_"`` creates ``coef_point_``.
        ``name="cluster_centers_"`` creates ``cluster_centers_point_``.
        """
        if not name.endswith("_"):
            raise ValueError(
                f"Expected fitted sklearn attribute ending in '_', got {name!r}."
            )

        alias = f"{name[:-1]}{self._reshaped_attr_suffix}"
        setattr(self, alias, value)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name.startswith("_"):
            return attr

        method_names = super().__getattribute__("_reshape_input_methods")
        if name not in method_names or not callable(attr):
            return attr

        @wraps(attr)
        def wrapped(X, *args, **kwargs):
            out = attr(self._flatten_X(X), *args, **kwargs)

            if name == "fit":
                self._reshape_attrs()

            return out

        return wrapped
