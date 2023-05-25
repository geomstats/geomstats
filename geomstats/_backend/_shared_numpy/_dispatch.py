from geomstats._backend import BACKEND_NAME

if BACKEND_NAME == "autograd":
    from autograd import numpy, scipy

    from ..autograd import _common

else:
    import numpy
    import scipy

    from ..numpy import _common
