import random

from polpo.testing.data import LazyCaseData
from polpo.testing.geometric import vectorize_datum

import geomstats.backend as gs


class VectorizableCaseData(LazyCaseData):
    """Case data whose vectorization data repeats the inputs of given cases.

    Subclasses implement ``_get_expected_func``.

    Parameters
    ----------
    rep_counts : list[int or tuple[int]]
        Batch shapes used to repeat the inputs of vectorization cases. An int
        is a single batch dimension. By default, a batch of one, one random
        single batch dimension and one random double batch dimension.
    **kwargs
        Keyword arguments of ``LazyCaseData``.
    """

    def __init__(self, rep_counts=None, **kwargs):
        super().__init__(**kwargs)

        if rep_counts is None:
            rep_counts = [
                1,
                random.randint(2, 5),
                (random.randint(2, 5), random.randint(2, 5)),
            ]

        self.rep_counts = [
            (rep_count,) if isinstance(rep_count, int) else tuple(rep_count)
            for rep_count in rep_counts
        ]

    def get_data_methods(self):
        return self._get_data_methods(excluded_suffixes="_vec_test_data")

    def get_vectorization_data_methods(self):
        return self._get_data_methods(suffix="_vec_test_data")

    def _get_expected_func(self, case):
        """Get the function computing the expected values of a case.

        Parameters
        ----------
        case : dict
            Case.

        Returns
        -------
        expected_func : callable
            Function of the unrepeated inputs, by name.
        """
        raise NotImplementedError

    def _vectorize(self, cases, expected_name="expected", get_expected_func=None):
        """Generate vectorization data comparing batched and single outputs.

        Parameters
        ----------
        cases : list[dict]
            Cases whose arrays are the inputs, which are repeated, and whose
            other values are passed unrepeated. A case may set a
            ``vectorization_type`` (by default ``"sym"``) and opt out from
            multiple batch dimensions with ``multi_batch=False``.
        expected_name : str or tuple[str]
            Name of the expected value. If a tuple, the expected function
            returns a tuple and each element gets the corresponding name.
        get_expected_func : callable
            Function of a case returning the function computing its expected
            values. By default, ``_get_expected_func``.

        Returns
        -------
        data : list[dict]
            Vectorization data, one for each batch shape of ``rep_counts``,
            whose expected values are computed from the unrepeated inputs.
        """
        if get_expected_func is None:
            get_expected_func = self._get_expected_func

        data = []
        for case in cases:
            case = case.copy()
            vectorization_type = case.pop("vectorization_type", "sym")
            multi_batch = case.pop("multi_batch", True)

            inputs = {name: value for name, value in case.items() if gs.is_array(value)}
            values = {
                name: value for name, value in case.items() if name not in inputs
            }
            expected_func = get_expected_func(case)
            for n_reps in self.rep_counts:
                if not multi_batch and len(n_reps) > 1:
                    continue

                data += vectorize_datum(
                    inputs,
                    expected_func,
                    expected_name=expected_name,
                    vectorization_type=vectorization_type,
                    n_reps=n_reps,
                    **values,
                )
        return data
