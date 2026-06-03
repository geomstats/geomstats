import numpy as np
from scipy._lib._array_api import array_namespace, xp_capabilities
from scipy.interpolate._cubic import CubicHermiteSpline, prepare_input


@xp_capabilities(
    cpu_only=True,
    jax_jit=False,
    skip_backends=[
        ("dask.array", "https://github.com/data-apis/array-api-extra/issues/488")
    ],
)
class PchipInterpolator(CubicHermiteSpline):
    r"""PCHIP shape-preserving interpolator (C1 smooth).

    ``x`` and ``y`` are arrays of values used to approximate some function f,
    with ``y = f(x)``. The interpolant uses monotonic cubic splines
    to find the value of new points. (PCHIP stands for Piecewise Cubic
    Hermite Interpolating Polynomial).

    Parameters
    ----------
    x : ndarray, shape (npoints, )
        A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
    y : ndarray, shape (..., npoints, ...)
        An N-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. Use the ``axis``
        parameter to select the interpolation axis.
    axis : int, optional
        Axis in the ``y`` array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    solve
    roots


    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    Akima1DInterpolator : Akima 1D interpolator.
    CubicSpline : Cubic spline data interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints.

    Notes
    -----
    The interpolator preserves monotonicity in the interpolation data and does
    not overshoot if the data is not smooth.

    The first derivatives are guaranteed to be continuous, but the second
    derivatives may jump at :math:`x_k`.

    Determines the derivatives at the points :math:`x_k`, :math:`f'_k`,
    by using PCHIP algorithm [1]_.

    Let :math:`h_k = x_{k+1} - x_k`, and  :math:`d_k = (y_{k+1} - y_k) / h_k`
    are the slopes at internal points :math:`x_k`.
    If the signs of :math:`d_k` and :math:`d_{k-1}` are different or either of
    them equals zero, then :math:`f'_k = 0`. Otherwise, it is given by the
    weighted harmonic mean

    .. math::

        \frac{w_1 + w_2}{f'_k} = \frac{w_1}{d_{k-1}} + \frac{w_2}{d_k}

    where :math:`w_1 = 2 h_k + h_{k-1}` and :math:`w_2 = h_k + 2 h_{k-1}`.

    The end slopes are set using a one-sided scheme [2]_.


    References
    ----------
    .. [1] F. N. Fritsch and J. Butland,
           A method for constructing local
           monotone piecewise cubic interpolants,
           SIAM J. Sci. Comput., 5(2), 300-304 (1984).
           :doi:`10.1137/0905021`.
    .. [2] C. Moler, Numerical Computing with Matlab, 2004.
           :doi:`10.1137/1.9780898717952`

    """

    # PchipInterpolator is not generic in scipy-stubs
    __class_getitem__ = None

    def __init__(self, x, y, axis=0, extrapolate=None):
        xp = array_namespace(x, y)
        x, _, y, axis, _ = prepare_input(x, y, axis, xp=xp)
        if xp.isdtype(y.dtype, "complex floating"):
            msg = (
                "`PchipInterpolator` only works with real values for `y`. "
                "If you are trying to use the real components of the passed array, "
                "use `np.real` on the array before passing to `PchipInterpolator`."
            )
            raise ValueError(msg)
        xv = xp.reshape(x, (x.shape[0],) + (1,) * (y.ndim - 1))
        dk = self._find_derivatives(xv, y, xp=xp)
        super().__init__(x, y, dk, axis=0, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _edge_case(h0, h1, m0, m1, xp):
        # one-sided three-point estimate for the derivative
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

        # try to preserve shape
        mask = xp.sign(d) != xp.sign(m0)
        mask2 = (xp.sign(m0) != xp.sign(m1)) & (xp.abs(d) > 3.0 * xp.abs(m0))
        mmm = (~mask) & mask2

        d[mask] = 0.0
        d[mmm] = 3.0 * m0[mmm]

        return d

    @staticmethod
    def _find_derivatives(x, y, xp):
        # Determine the derivatives at the points y_k, d_k, by using
        #  PCHIP algorithm is:
        # We choose the derivatives at the point x_k by
        # Let m_k be the slope of the kth segment (between k and k+1)
        # If m_k=0 or m_{k-1}=0 or sgn(m_k) != sgn(m_{k-1}) then d_k == 0
        # else use weighted harmonic mean:
        #   w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        #   1/d_k = 1/(w_1 + w_2)*(w_1 / m_k + w_2 / m_{k-1})
        #   where h_k is the spacing between x_k and x_{k+1}
        y_shape = y.shape
        if y.ndim == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            y = y[:, None]

        hk = x[1:] - x[:-1]
        mk = (y[1:] - y[:-1]) / hk

        if y.shape[0] == 2:
            # edge case: only have two points, use linear interpolation
            dk = xp.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return xp.reshape(dk, y_shape)

        smk = xp.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)

        w1 = 2 * hk[1:] + hk[:-1]
        w2 = hk[1:] + 2 * hk[:-1]

        # values where division by zero occurs will be excluded
        # by 'condition' afterwards
        with np.errstate(divide="ignore", invalid="ignore"):
            whmean = (w1 / mk[:-1] + w2 / mk[1:]) / (w1 + w2)

        # TODO (L): backend issue
        # dk = np.zeros_like(y)
        dk = xp.zeros_like(y)

        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean[~condition]

        # special case endpoints, as suggested in
        # Cleve Moler, Numerical Computing with MATLAB, Chap 3.6 (pchiptx.m)
        dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1], xp=xp)
        dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2], xp=xp)

        return xp.reshape(dk, y_shape)
