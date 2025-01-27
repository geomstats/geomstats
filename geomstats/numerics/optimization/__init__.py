try:
    from ._torch import TorchLBFGS
except ImportError:
    pass

try:
    from ._torchmin import TorchminMinimize
except ImportError:
    pass

from ._optimization import Minimizer, NewtonMethod, RootFinder
from ._scipy import ScipyMinimize, ScipyRoot
