def value_and_grad(objective):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend."
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend through: export GEOMSTATS_BACKEND=autograd")


def jacobian(f):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend."
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend through: export GEOMSTATS_BACKEND=autograd")
