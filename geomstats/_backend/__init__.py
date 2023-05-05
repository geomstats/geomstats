"""Execution backends.

Lead authors: Johan Mathe and Niklas Koep.
"""

import importlib
import logging
import os
import sys
import types

import geomstats._backend._common as common

BACKEND_ATTRIBUTES = {
    "": [
        # Types
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
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
        "as_dtype",
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
        "gamma",
        "get_default_dtype",
        "get_default_cdtype",
        "get_slice",
        "greater",
        "hsplit",
        "hstack",
        "imag",
        "isclose",
        "isnan",
        "is_array",
        "is_complex",
        "is_floating",
        "is_bool",
        "kron",
        "less",
        "less_equal",
        "linspace",
        "log",
        "logical_and",
        "logical_or",
        "mat_from_diag_triu_tril",
        "matmul",
        "matvec",
        "maximum",
        "mean",
        "meshgrid",
        "minimum",
        "mod",
        "moveaxis",
        "ndim",
        "one_hot",
        "ones",
        "ones_like",
        "outer",
        "pad",
        "pi",
        "polygamma",
        "power",
        "prod",
        "quantile",
        "ravel_tril_indices",
        "real",
        "repeat",
        "reshape",
        "rtol",
        "scatter_add",
        "searchsorted",
        "set_default_dtype",
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
        "take",
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
        "trapz",
    ],
    "autodiff": [
        "custom_gradient",
        "hessian",
        "hessian_vec",
        "jacobian",
        "jacobian_vec",
        "jacobian_and_hessian",
        "value_and_grad",
        "value_jacobian_and_hessian",
    ],
    "linalg": [
        "cholesky",
        "det",
        "eig",
        "eigh",
        "eigvalsh",
        "expm",
        "fractional_matrix_power",
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
        try:
            return importlib.import_module(f"geomstats._backend.{backend_name}")
        except ModuleNotFoundError:
            raise RuntimeError(f"Unknown backend '{backend_name}'")

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
                        f"Backend '{backend_name}' exposes no '{module_name}' module"
                    ) from None
                new_submodule = types.ModuleType(f"{self._path}.{module_name}")
                new_submodule.__file__ = submodule.__file__
                setattr(new_module, module_name, new_submodule)
            else:
                submodule = backend
                new_submodule = new_module
            for attribute_name in attributes:
                try:
                    submodule_ = submodule
                    if module_name == "" and not hasattr(submodule, attribute_name):
                        submodule_ = common
                    attribute = getattr(submodule_, attribute_name)

                except AttributeError:
                    if module_name:
                        error = (
                            f"Module '{module_name}' of backend '{backend_name}' "
                            f"has no attribute '{attribute_name}'"
                        )
                    else:
                        error = (
                            f"Backend '{backend_name}' has no "
                            f"attribute '{attribute_name}'"
                        )

                    raise RuntimeError(error) from None
                else:
                    setattr(new_submodule, attribute_name, attribute)

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
        module.__name__ = f"geomstats.{_BACKEND}"
        module.__loader__ = self
        sys.modules[fullname] = module

        module.set_default_dtype("float64")

        logging.info(f"Using {_BACKEND} backend")
        return module


sys.meta_path.append(BackendImporter("geomstats.backend"))
