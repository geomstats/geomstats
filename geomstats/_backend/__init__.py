"""Execution backends.

Lead authors: Johan Mathe and Niklas Koep.
"""

import logging
import os
import sys
import types

BACKEND_ATTRIBUTES = {
    "": [
        # Types
        "int32",
        "int64",
        "float32",
        "float64",
        "uint8",
        # Functions
        "abs",
        "all",
        "allclose",
        "amax",
        "amin",
        "angle",
        "any",
        "arange",
        "arccos",
        "arccosh",
        "arcsin",
        "arctan2",
        "arctanh",
        "argmax",
        "argmin",
        "array",
        "array_from_sparse",
        "assignment",
        "assignment_by_sum",
        "atol",
        "broadcast_arrays",
        "broadcast_to",
        "cast",
        "ceil",
        "clip",
        "comb",
        "concatenate",
        "conj",
        "convert_to_wider_dtype",
        "copy",
        "cos",
        "cosh",
        "cross",
        "cumprod",
        "cumsum",
        "diag_indices",
        "diagonal",
        "divide",
        "dot",
        "einsum",
        "empty",
        "empty_like",
        "equal",
        "erf",
        "exp",
        "expand_dims",
        "eye",
        "flatten",
        "flip",
        "floor",
        "from_numpy",
        "get_mask_i_float",
        "get_slice",
        "greater",
        "hsplit",
        "hstack",
        "imag",
        "isclose",
        "isnan",
        "kron",
        "less",
        "less_equal",
        "linspace",
        "log",
        "logical_and",
        "logical_or",
        "mat_from_diag_triu_tril",
        "matmul",
        "maximum",
        "mean",
        "meshgrid",
        "minimum",
        "mod",
        "ndim",
        "one_hot",
        "ones",
        "ones_like",
        "outer",
        "polygamma",
        "power",
        "prod",
        "ravel_tril_indices",
        "real",
        "repeat",
        "reshape",
        "rtol",
        "searchsorted",
        "set_diag",
        "shape",
        "sign",
        "sin",
        "sinh",
        "split",
        "sqrt",
        "squeeze",
        "sort",
        "stack",
        "std",
        "sum",
        "tan",
        "tanh",
        "tile",
        "to_numpy",
        "to_ndarray",
        "trace",
        "transpose",
        "tril",
        "triu",
        "tril_indices",
        "triu_indices",
        "tril_to_vec",
        "triu_to_vec",
        "vec_to_diag",
        "unique",
        "vectorize",
        "vstack",
        "where",
        "zeros",
        "zeros_like",
    ],
    "autodiff": ["custom_gradient", "detach", "jacobian", "value_and_grad"],
    "linalg": [
        "cholesky",
        "det",
        "eig",
        "eigh",
        "eigvalsh",
        "expm",
        "inv",
        "is_single_matrix_pd",
        "logm",
        "norm",
        "qr",
        "quadratic_assignment",
        "solve",
        "solve_sylvester",
        "sqrtm",
        "svd",
        "matrix_rank",
    ],
    "random": [
        "choice",
        "normal",
        "multivariate_normal",
        # TODO (nkoep): Remove 'rand' and replace it by 'uniform'. Much like
        #              'randn' is a convenience wrapper (which we don't use)
        #              for 'normal', 'rand' only wraps 'uniform'.
        "rand",
        "randint",
        "seed",
        "uniform",
    ],
}


class BackendImporter:
    """Importer class to create the backend module."""

    def __init__(self, path):
        self._path = path

    @staticmethod
    def _import_backend(backend_name):
        if backend_name == "autograd":
            from geomstats._backend import autograd as backend
        elif backend_name == "numpy":
            from geomstats._backend import numpy as backend
        elif backend_name == "pytorch":
            from geomstats._backend import pytorch as backend
        elif backend_name == "tensorflow":
            from geomstats._backend import tensorflow as backend
        else:
            raise RuntimeError("Unknown backend '{:s}'".format(backend_name))
        return backend

    def _create_backend_module(self, backend_name):
        backend = self._import_backend(backend_name)

        new_module = types.ModuleType(self._path)
        new_module.__file__ = backend.__file__

        for module_name, attributes in BACKEND_ATTRIBUTES.items():
            if module_name:
                try:
                    submodule = getattr(backend, module_name)
                except AttributeError:
                    raise RuntimeError(
                        "Backend '{}' exposes no '{}' module".format(
                            backend_name, module_name
                        )
                    ) from None
                new_submodule = types.ModuleType(
                    "{}.{}".format(self._path, module_name)
                )
                new_submodule.__file__ = submodule.__file__
                setattr(new_module, module_name, new_submodule)
            else:
                submodule = backend
                new_submodule = new_module
            for attribute_name in attributes:
                try:
                    attribute = getattr(submodule, attribute_name)
                except AttributeError:
                    if module_name:
                        error = (
                            "Module '{}' of backend '{}' has no "
                            "attribute '{}'".format(
                                module_name, backend_name, attribute_name
                            )
                        )
                    else:
                        error = "Backend '{}' has no attribute '{}'".format(
                            backend_name, attribute_name
                        )
                    raise RuntimeError(error) from None
                else:
                    setattr(new_submodule, attribute_name, attribute)

        from numpy import pi

        new_module.pi = pi

        return new_module

    def find_module(self, fullname, path=None):
        """Find module."""
        if self._path != fullname:
            return None
        return self

    def load_module(self, fullname):
        """Load module."""
        if fullname in sys.modules:
            return sys.modules[fullname]

        _BACKEND = os.environ.get("GEOMSTATS_BACKEND")
        if _BACKEND is None:
            os.environ["GEOMSTATS_BACKEND"] = _BACKEND = "numpy"

        module = self._create_backend_module(_BACKEND)
        module.__loader__ = self
        sys.modules[fullname] = module

        logging.info("Using {:s} backend".format(_BACKEND))
        return module


sys.meta_path.append(BackendImporter("geomstats.backend"))
