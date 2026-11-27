"""Logging helpers."""


def warn_max_iterations(logger, max_iter):
    """Log a warning that the maximum number of iterations was reached.

    Callers decide when the iteration budget is exhausted.

    Parameters
    ----------
    logger : logging.Logger
        Logger of the calling module.
    max_iter : int
        Maximum number of iterations.
    """
    logger.warning(
        "Maximum number of iterations %d reached. The result may be inaccurate.",
        max_iter,
    )
