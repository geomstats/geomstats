import numpy as np

import geomstats.backend as gs


def _convert_gs_to_np(value):
    if gs.is_array(value):
        return gs.to_numpy(value)

    if isinstance(value, (list, tuple)):
        new_value = []
        for value_ in value:
            new_value.append(_convert_gs_to_np(value_))

        if isinstance(value, tuple):
            new_value = tuple(new_value)

        return new_value

    if isinstance(value, dict):
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


def get_backend_fncs(func_name, cmp_package=np):
    func_name_ls = func_name.split(".")
    gs_func = gs
    cmp_func = cmp_package
    for name in func_name_ls:
        gs_func = getattr(gs_func, name)
        cmp_func = getattr(cmp_func, name)

    return gs_func, cmp_func


def get_backend_fnc(func_name):
    func_name_ls = func_name.split(".")
    gs_func = gs
    for name in func_name_ls:
        gs_func = getattr(gs_func, name)

    return gs_func
