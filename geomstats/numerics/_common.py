import numpy as np

import geomstats.backend as gs


def result_to_backend_type(ode_result):
    if gs.__name__.endswith("numpy"):
        return ode_result

    for key, value in ode_result.items():
        if type(value) is np.ndarray:
            ode_result[key] = gs.array(value)

    return ode_result
