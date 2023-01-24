import copy
import itertools

from geomstats.errors import check_parameter_accepted_values
from geomstats.vectorization import repeat_point


def _filter_combs(combs, vec_type):
    MAP_VEC_TYPE = {
        "repeat-first": 1,
        "repeat-second": 0,
    }
    index = MAP_VEC_TYPE[vec_type]
    other_index = (index + 1) % 2

    for comb in combs.copy():
        if comb[index] == 1 and comb[index] != comb[other_index]:
            combs.remove(comb)

    return combs


def _generate_datum_vectorization_data(
    datum, comb_indices, arg_names, expected_name, n_reps=2
):

    if expected_name is not None:
        has_expected = True
        expected = datum.get(expected_name)
        expected_rep = repeat_point(expected, n_reps=n_reps)
    else:
        has_expected = False

    args_combs = []
    for arg_name in arg_names:
        arg = datum.get(arg_name)
        args_combs.append([arg, repeat_point(arg, n_reps=n_reps, expand=True)])

    new_data = []
    for indices in comb_indices:
        new_datum = copy.copy(datum)

        if has_expected:
            new_datum[expected_name] = expected_rep if 1 in indices else expected

        for arg_i, (index, arg_name) in enumerate(zip(indices, arg_names)):
            new_datum[arg_name] = args_combs[arg_i][index]

        new_data.append(new_datum)

    return new_data


def generate_vectorization_data(
    data,
    arg_names,
    expected_name=None,
    n_reps=2,
    vectorization_type="sym",
):
    """Create new data with vectorized version of inputs.

    Parameters
    ----------
    data : list of dict
        Data. Each to vectorize.
    arg_names: list
        Name of inputs to vectorize.
    expected_name: str
        Output name in case it needs to be repeated.
    n_reps: int
        Number of times the input points should be repeated.
    vectorization_type: str
        Possible values are `sym`, `repeat-first`, `repeat-second`.
        `repeat-first` and `repeat-second` only valid for two argument case.
        `repeat-first` and `repeat-second` test asymmetric cases, repeating
        only first or second input, respectively.
    """
    check_parameter_accepted_values(
        vectorization_type,
        "vectorization_type",
        ["sym", "repeat-first", "repeat-second"],
    )

    n_args = len(arg_names)
    if n_args != 2 and vectorization_type != "sym":
        raise NotImplementedError(
            f"`{vectorization_type} only implemented for 2 arguments."
        )

    comb_indices = list(itertools.product(*[range(2)] * n_args))
    if n_args == 2 and vectorization_type != "sym":
        comb_indices = _filter_combs(comb_indices, vectorization_type)

    new_data = []
    for datum in data:
        new_data.extend(
            _generate_datum_vectorization_data(
                datum,
                comb_indices,
                arg_names,
                expected_name=expected_name,
                n_reps=n_reps,
            )
        )

    return new_data
