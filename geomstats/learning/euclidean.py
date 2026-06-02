from ._sklearn import EuclideanInputMixin, _enable_array_dispatch
from ._sklearngs.linear_model._base import LinearRegression as _LinearRegression

_enable_array_dispatch()


# TODO: tests; y 1d/2d, X 1d/2d


class LinearRegression(EuclideanInputMixin, _LinearRegression):
    """Linear regression accepting structured Euclidean inputs."""

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

    def _reshape_attrs(self):
        coef = self.coef_
        coef = coef.reshape((*coef.shape[:-1], *self.space.shape))

        self._set_reshaped_attr("coef_", coef)
