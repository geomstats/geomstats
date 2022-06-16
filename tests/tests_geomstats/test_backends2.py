import numpy as np
import pytest

import geomstats.backend as gs
from tests.conftest import Parametrizer, TestCase
from tests.data.backends_data import BackendsTestData


def _convert_gs_to_np(value):
    if gs.is_array(value):
        return gs.to_numpy(value)

    elif isinstance(value, (list, tuple)):
        new_value = []
        for value_ in value:
            new_value.append(_convert_gs_to_np(value_))

        if isinstance(value, tuple):
            new_value = tuple(new_value)

        return new_value

    elif isinstance(value, dict):
        return {key: _convert_gs_to_np(value_) for key, value_ in value.items()}

    return value


def convert_gs_to_np(*args):
    out = []
    for arg in args:
        val = _convert_gs_to_np(arg)
        out.append(val)

    if len(args) == 1:
        return out[0]

    return out


def get_backend_fncs(func_name, numpy=True):
    gs_fnc = getattr(gs, func_name)
    if not numpy:
        return gs_fnc

    return gs_fnc, getattr(np, func_name)


class TestBackends(TestCase, metaclass=Parametrizer):
    testing_data = BackendsTestData()

    def test_np_like_array(self, func_name, args):
        # TODO: skip for numpy

        gs_fnc, np_fnc = get_backend_fncs(func_name)
        np_args = convert_gs_to_np(args)

        gs_array = gs_fnc(*args)
        np_array = np_fnc(*np_args)

        self.assertAllCloseToNp(gs_array, np_array)

    def test_np_like_binary_op(self, func_name, a, b):
        gs_fnc, np_fnc = get_backend_fncs(func_name)
        np_a, np_b = convert_gs_to_np(a, b)

        gs_out = gs_fnc(a, b)
        np_out = np_fnc(np_a, np_b)
        self.assertAllCloseToNp(gs_out, np_out)

    def test_binary_op_vec(self, func_name, a, b):
        gs_fnc = get_backend_fncs(func_name, numpy=False)

        res = gs_fnc(a, b)

        rep_a = gs.repeat(gs.expand_dims(a, 0), 2, axis=0)
        rep_b = gs.repeat(gs.expand_dims(b, 0), 2, axis=0)

        res_rep_a = gs_fnc(rep_a, b)
        res_rep_b = gs_fnc(a, rep_b)
        res_rep_a_b = gs_fnc(rep_a, rep_b)

        self.assertAllClose(res_rep_a, res_rep_b)
        self.assertAllClose(res_rep_a, res_rep_a_b)
        for res_ in res_rep_a_b:
            self.assertAllClose(res_, res)

    def test_binary_raises_error(self, func_name, a, b):
        gs_fnc = get_backend_fncs(func_name, numpy=False)

        with pytest.raises(Exception):
            gs_fnc(a, b)
