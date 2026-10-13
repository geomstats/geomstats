from ._optimization import Minimizer, NewtonMethod, RootFinder
from ._scipy import ScipyMinimize, ScipyRoot

__all__ = ["Minimizer", "NewtonMethod", "RootFinder", "ScipyMinimize", "ScipyRoot"]

try:
    from ._torch import TorchAdam, TorchLBFGS, TorchRMSprop, TorchSGD
except ImportError:
    pass
else:
    __all__.extend(["TorchAdam", "TorchLBFGS", "TorchRMSprop", "TorchSGD"])

try:
    from ._torchmin import TorchminMinimize
except ImportError:
    pass
else:
    __all__.append("TorchminMinimize")
