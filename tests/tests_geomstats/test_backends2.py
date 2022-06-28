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


def get_backend_fncs(func_name):
    return getattr(gs, func_name), getattr(np, func_name)


class TestBackends(TestCase, metaclass=Parametrizer):
    testing_data = BackendsTestData()

    def test_array_like_np(self, func_name, args):
        # TODO: skip for numpy?

        gs_fnc, np_fnc = get_backend_fncs(func_name)
        np_args = convert_gs_to_np(args)

        gs_array = gs_fnc(*args)
        np_array = np_fnc(*np_args)

        self.assertAllCloseToNp(gs_array, np_array)

    def test_binary_op_like_np(self, func_name, a, b):
        gs_fnc, np_fnc = get_backend_fncs(func_name)
        np_a, np_b = convert_gs_to_np(a, b)

        gs_out = gs_fnc(a, b)
        np_out = np_fnc(np_a, np_b)
        self.assertAllCloseToNp(gs_out, np_out)

    def test_binary_op_like_einsum(self, func_name, a, b, einsum_expr):
        gs_fnc = getattr(gs, func_name)

        gs_out = gs_fnc(a, b)
        ein_out = gs.einsum(einsum_expr, a, b)

        self.assertAllClose(gs_out, ein_out)

    def test_binary_op_vec(self, func_name, a, b):
        gs_fnc = getattr(gs, func_name)

        res = gs_fnc(a, b)

        a_expanded = gs.expand_dims(a, 0)
        b_expanded = gs.expand_dims(b, 0)

        a_rep = gs.repeat(a_expanded, 2, axis=0)
        b_rep = gs.repeat(b_expanded, 2, axis=0)

        res_a_rep = gs_fnc(a_rep, b)
        res_b_rep = gs_fnc(a, b_rep)
        res_a_b_rep = gs_fnc(a_rep, b_rep)
        res_a_expanded = gs_fnc(a_expanded, b_rep)
        res_b_expanded = gs_fnc(a_rep, b_expanded)

        self.assertAllClose(res_a_rep, res_a_b_rep)
        self.assertAllClose(res_b_rep, res_a_b_rep)
        self.assertAllClose(res_a_expanded, res_a_b_rep)
        self.assertAllClose(res_b_expanded, res_a_b_rep)
        for res_ in res_a_b_rep:
            self.assertAllClose(res_, res)

    def test_binary_op_vec_raises_error(self, func_name, a, b):
        a_rep = gs.repeat(gs.expand_dims(a, 0), 2, axis=0)
        b_rep = gs.repeat(gs.expand_dims(b, 0), 3, axis=0)

        self.test_binary_op_raises_error(func_name, a_rep, b_rep)

    def test_binary_op_raises_error(self, func_name, a, b):
        gs_fnc = getattr(gs, func_name)

        with pytest.raises(Exception):
            gs_fnc(a, b)

    def test_binary_op_runs(self, func_name, a, b):
        gs_fnc = getattr(gs, func_name)
        gs_fnc(a, b)

    def test_bool_unary_func(self, func_name, a, expected):
        gs_fnc = getattr(gs, func_name)

        out = gs_fnc(a)
        if expected:
            self.assertTrue(out)
        else:
            self.assertFalse(out)

    def test_func_out_shape(self, func_name, args, expected):
        gs_fnc = getattr(gs, func_name)

        # TODO: better comparison (for more info when failing)
        out = gs_fnc(*args)
        self.assertTrue(gs.shape(out) == expected)
