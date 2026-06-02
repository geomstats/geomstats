from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from sklearn.base import (
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.linear_model._base import LinearModel, _preprocess_data
from sklearn.utils._param_validation import Interval
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    validate_data,
)

import geomstats.backend as gs


class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    tol : float, default=1e-6
        The precision of the solution (`coef_`) is determined by `tol` which
        specifies a different convergence criterion for the `lsqr` solver.
        `tol` is set as `atol` and `btol` of :func:`scipy.sparse.linalg.lsqr` when
        fitting on sparse training data. This parameter has no effect when fitting
        on dense data.

        .. versionadded:: 1.7

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        For a comparison between a linear regression model with positive constraints
        on the regression coefficients and a linear regression without such constraints,
        see :ref:`sphx_glr_auto_examples_linear_model_plot_nnls.py`.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (:func:`scipy.linalg.lstsq`) or Non Negative Least Squares
    (:func:`scipy.optimize.nnls`) wrapped as a predictor object.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    np.float64(3.0)
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "n_jobs": [None, Integral],
        "positive": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        tol=1e-6,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=accept_sparse,
            y_numeric=True,
            multi_output=True,
            force_writeable=True,
        )

        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype, ensure_non_negative=True
            )

        # Note that neither _rescale_data nor the rest of the fit method of
        # LinearRegression can benefit from in-place operations when X is a
        # sparse matrix. Therefore, let's not copy X when it is sparse.
        copy_X_in_preprocess_data = self.copy_X and not sp.issparse(X)

        X, y, X_offset, y_offset, _, sample_weight_sqrt = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=copy_X_in_preprocess_data,
            sample_weight=sample_weight,
        )

        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        elif sp.issparse(X):
            if has_sw:

                def matvec(b):
                    return X.dot(b) - sample_weight_sqrt * b.dot(X_offset)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset * b.dot(sample_weight_sqrt)

            else:

                def matvec(b):
                    return X.dot(b) - b.dot(X_offset)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset * b.sum()

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y, atol=self.tol, btol=self.tol)[0]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsqr)(
                        X_centered, y[:, j].ravel(), atol=self.tol, btol=self.tol
                    )
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            # TODO (L): backend issue
            if gs.__name__.endswith("pytorch"):
                # TODO: update geomstats.backend
                import torch

                cond = max(X.shape) * torch.finfo(X.dtype).eps
                lstsq = torch.linalg.lstsq(X, y, rcond=cond)
                self.coef_ = lstsq.solution
                self.rank_ = lstsq.rank
                self.singular_ = lstsq.singular_values

            else:
                # cut-off ratio for small singular values
                cond = max(X.shape) * np.finfo(X.dtype).eps
                self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(
                    X, y, cond=cond
                )

            self.coef_ = self.coef_.T

        if y.ndim == 1:
            # TODO (L): backend issue
            # self.coef_ = np.ravel(self.coef_)
            self.coef_ = gs.reshape(self.coef_, (-1,))
        self._set_intercept(X_offset, y_offset)
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = not self.positive
        return tags
