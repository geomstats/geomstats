import copy
import itertools

from geomstats.vectorization import repeat_point


def _filter_combs(n_args, combs, vectorization_type):
    if vectorization_type == "sym" or n_args == 1:
        return combs

    repeat_indices = [int(val) for val in vectorization_type.split("-")[1:]]
    remove_indices = [val for val in range(n_args) if val not in repeat_indices]

    for comb in combs.copy():
        vals_remove = [comb[remove_index] for remove_index in remove_indices]
        if max(vals_remove) == 0:
            continue

        elif min(vals_remove) == 1 and min(comb) == 1:
            continue

        combs.remove(comb)

    return combs


def _check_vectorization_type(vectorization_type, n_args):
    if vectorization_type == "sym" or n_args == 1:
        return

    try:
        for val in vectorization_type.split("-")[1:]:
            int(val)

        if int(val) > n_args:
            raise ValueError(
                "Unable to repeat unexisting args for vectorization type "
                f"`{vectorization_type}` and `n_args = {n_args}"
            )

    except ValueError:
        raise ValueError(
            f"Unable to understand vectorization type `{vectorization_type}`. "
            "Can handle `sym` and `repeat-(int)` format."
        )


def _generate_datum_vectorization_data(
    datum, comb_indices, arg_names, expected_name, n_reps=2
):
    if expected_name is not None:
        has_expected = True
        if type(expected_name) is str:
            expected_name = [expected_name]

        expected_combs = []
        for expected_name_ in expected_name:
            expected = datum.get(expected_name_)
            expected_combs.append([expected, repeat_point(expected, n_reps=n_reps)])
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
            rep = int(1 in indices)
            for expected_i, expected_name_ in enumerate(expected_name):
                new_datum[expected_name_] = expected_combs[expected_i][rep]

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
    expected_name: str or list of str
        Output name in case it needs to be repeated.
    n_reps: int
        Number of times the input points should be repeated.
    vectorization_type: str
        Possible values are `sym` or the format `repeat-(int)` (e.g. "repeat-0-2").
    """
    n_args = len(arg_names)
    _check_vectorization_type(vectorization_type, n_args)

    comb_indices = list(itertools.product(*[range(2)] * n_args))
    comb_indices = _filter_combs(n_args, comb_indices, vectorization_type)

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
