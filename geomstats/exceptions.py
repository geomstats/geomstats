"""Geomstats custom exceptions."""


class AutodiffNotImplementedError(RuntimeError):
    """Raised when autodiff is not implemented."""


class NotPartialOrder(Exception):
    """Raise an exception when less equal is not true."""
