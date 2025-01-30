try:
    from ._torch import TorchAdam, TorchLBFGS, TorchRMSprop, TorchSGD
except ImportError:
    pass

try:
    from ._torchmin import TorchminMinimize
except ImportError:
    pass

from ._optimization import Minimizer, NewtonMethod, RootFinder
from ._scipy import ScipyMinimize, ScipyRoot
