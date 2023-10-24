import copy
import inspect
import itertools

from geomstats.vectorization import repeat_point

KNOWN_POINT_NAMES = (
    "point",
    "base_point",
    "point_a",
    "point_b",
    "fiber_point",
    "mat",
)
KNOWN_VECTOR_NAMES = (
    "vec",
    "vector",
    "tangent_vec",
    "tangent_vec_a",
    "tangent_vec_b",
    "tangent_vec_c",
    "tangent_vec_d",
    "cotangent_vec_a",
    "cotangent_vec_b",
    "direction",
)
KNOWN_ARGS = KNOWN_POINT_NAMES + KNOWN_VECTOR_NAMES


def _filter_combs(n_args, combs, vectorization_type):
    if vectorization_type == "sym" or n_args == 1:
        return combs

    repeat_indices = [int(val) for val in vectorization_type.split("-")[1:]]
    if len(repeat_indices) == n_args:
        return combs

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
    if vectorization_type in ("sym", "basic") or n_args == 1:
        return

    try:
        val = None
        for val in vectorization_type.split("-")[1:]:
            int(val)

        if val is None or int(val) > n_args:
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
        Possible values are 'sym', 'basic', or the format 'repeat-(int)'
        (e.g. "repeat-0-2").
        'sym': tests all repetition combinations.
        'basic': tests only no repetition and repetition of all.
        'repeat-(int)': tests repetition of provided indices.
    """
    n_args = len(arg_names)
    _check_vectorization_type(vectorization_type, n_args)

    if vectorization_type == "basic":
        comb_indices = [tuple(i for _ in range(n_args)) for i in range(2)]
    else:
        comb_indices = list(itertools.product(*[range(2)] * n_args))
        comb_indices = _filter_combs(n_args, comb_indices, vectorization_type)
    comb_indices.pop(0)

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


def _generate_random_data(data_generator, arg_names):
    data = {}
    base_point = None
    for arg_name in arg_names:
        if arg_name in KNOWN_POINT_NAMES:
            base_point = data[arg_name] = data_generator.random_point()

    for arg_name in arg_names:
        if arg_name in KNOWN_VECTOR_NAMES:
            data[arg_name] = data_generator.random_tangent_vec(base_point)

    return data


def _get_vectorization_type(test_case, arg_names):
    if test_case.tangent_to_multiple:
        return "sym"

    tangent_vec_type = ""
    for k, arg_name in enumerate(arg_names):
        if arg_name in KNOWN_VECTOR_NAMES:
            tangent_vec_type += f"-{k}"

    return "repeat" + tangent_vec_type if tangent_vec_type else "sym"


def test_vectorization(self, test_func, n_reps, atol):
    # TODO: move this to decorator?
    # TODO: accept kwargs?
    arg_names = list(inspect.signature(test_func).parameters.keys())
    arg_names = list(filter(lambda x: x in KNOWN_ARGS, arg_names))
    data = _generate_random_data(self.data_generator, arg_names)

    geometry = (
        self.space.metric
        if hasattr(self, "is_metric") and self.is_metric
        else self.space
    )
    data["expected"] = getattr(geometry, test_func.__name__[5:])(**data)
    data["atol"] = atol

    vec_data = generate_vectorization_data(
        data=[data],
        arg_names=arg_names,
        expected_name="expected",
        n_reps=n_reps,
        vectorization_type=_get_vectorization_type(self, arg_names),
    )
    self._test_vectorization(vec_data, test_fnc_name=test_func.__name__)
