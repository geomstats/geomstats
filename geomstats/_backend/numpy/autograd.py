def value_and_grad(objective):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command export GEOMSTATS_BACKEND=autograd in a terminal.")


def jacobian(f):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command export GEOMSTATS_BACKEND=autograd in a terminal.")
