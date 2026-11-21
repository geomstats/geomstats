import numpy as np

from geomstats.test.test_case import TestCase
from geomstats.test_cases.backend._utils import convert_gs_to_np, get_backend_fncs


class AgainstNumpyTestCase(TestCase):
    def test_func_like_np(self, func_name, args):
        """Check a function against numpy.

        Parameters
        ----------
        func_name : str
            Name of the backend function, e.g. ``"linalg.matrix_rank"``.
        args : tuple
            Positional arguments of the function.
        """
        gs_fnc, np_fnc = get_backend_fncs(func_name, cmp_package=np)

        gs_array = gs_fnc(*args)
        np_array = np_fnc(*convert_gs_to_np(args))

        self.assertAllCloseToNp(gs_array, np_array)
