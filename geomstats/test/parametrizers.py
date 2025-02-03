import functools
import importlib
import inspect
import json
import os
import shutil
import tempfile
import types
import warnings

import nbformat
import pytest
from matplotlib import pyplot as plt
from nbconvert.preprocessors import ExecutePreprocessor

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
from geomstats.test.test_case import autodiff_backend
from geomstats.test.vectorization import test_vectorization

NAME2MARK = {
    "xfail": pytest.mark.xfail(),
    "vec": pytest.mark.vec(),
    "skip": pytest.mark.skip(),
}


class TestFunction:
    def __init__(self, name, func, active=False):
        self.name = name
        self.func = func
        self._marks = self._collect_initial_marks()
        self._missing_marks = []
        self.active = active
        self._arg_names = None
        self.tolerances = {}

    def _collect_initial_marks(self):
        if not hasattr(self.func, "pytestmark"):
            return []
        return _get_pytest_mark_names(self.func)

    @property
    def skip(self):
        return "skip" in self.marks

    @property
    def marks(self):
        return self._marks + self._missing_marks

    @property
    def vanilla_name(self):
        return self.name[5:]

    @property
    def data_name(self):
        return f"{self.vanilla_name}_test_data"

    def add_mark(self, mark_name):
        if mark_name not in self.marks:
            self._missing_marks.append(mark_name)

    @property
    def arg_names(self):
        if self._arg_names is None:
            self._arg_names = inspect.getfullargspec(self.func)[0]
            self._handle_tolerances_defaults()
        return self._arg_names

    @property
    def arg_str(self):
        return ", ".join(self.arg_names[1:])

    def _handle_tolerances_defaults(self):
        for arg_name in self.arg_names:
            if arg_name.endswith("rtol"):
                self.tolerances.setdefault(arg_name, gs.rtol)
            elif arg_name.endswith("tol"):
                self.tolerances.setdefault(arg_name, 1e-6)

    def collect(
        self, testing_data, decorators=(), conditional_decorators=(), skip_vec=False
    ):
        test_func, default_values = _copy_func(self.func)
        if not self.active:
            return pytest.mark.ignore()(test_func)

        if self.skip or skip_vec and "vec" in self.marks:
            return pytest.mark.skip()(test_func)

        for decorator in decorators:
            test_func = decorator(test_func)

        for condition_func, decorator in conditional_decorators:
            if condition_func(self):
                test_func = decorator(test_func)

        for mark_name in self._missing_marks:
            test_func = NAME2MARK[mark_name]()(test_func)

        # no args case (note selection was done above)
        if len(self.arg_names) == 1:
            return test_func

        test_data = self._get_test_data(testing_data, default_values)
        return pytest.mark.parametrize(self.arg_str, test_data)(test_func)

    def _get_test_data(self, testing_data, default_values):
        # assumes pairing test-data exists
        test_data = getattr(testing_data, self.data_name)()

        if test_data is None:
            raise Exception(f"'{self.data_name}' returned None. should be list")

        test_data = self._dictify_test_data(test_data)
        test_data = self._pytestify_test_data(test_data, default_values)

        return test_data

    def _dictify_test_data(self, test_data):
        test_data_ = []
        for test_datum in test_data:
            if not isinstance(test_datum, dict):
                test_datum = dict(zip(self.arg_names[1:], test_datum[:-1]))
                test_datum["marks"] = test_datum[-1]

            for tol_arg_name, tol in self.tolerances.items():
                test_datum.setdefault(tol_arg_name, tol)

            test_data_.append(test_datum)

        return test_data_

    def _pytestify_test_data(self, test_data, default_values):
        tests = []
        for test_datum in test_data:
            try:
                values = [
                    test_datum.get(key) if key in test_datum else default_values[key]
                    for key in self.arg_names[1:]
                ]

            except KeyError:
                raise Exception(
                    f"{self.name} requires the following arguments: "
                    f"{', '.join(self.arg_names[1:])}"
                )
            tests.append(pytest.param(*values, marks=test_datum.get("marks")))

        return tests


class AutoVecTestFunction(TestFunction):
    def __init__(self, associated_test_func):
        super().__init__(f"{associated_test_func.name}_vec", None)
        self.associated_test_func = associated_test_func
        self.active = True
        self.add_mark("vec")

    def collect(
        self, testing_data, decorators=(), conditional_decorators=(), skip_vec=False
    ):
        if self.active:
            self._generate_vectorized_func()
        return super().collect(
            testing_data,
            decorators=decorators,
            conditional_decorators=conditional_decorators,
            skip_vec=skip_vec,
        )

    def _generate_vectorized_func(self):
        test_func = self.associated_test_func.func

        def new_test(self, n_reps, atol):
            return test_vectorization(self, test_func, n_reps, atol)

        self.func = new_test


def _update_attrs(test_funcs, testing_data, attrs):
    decorators = _collect_decorators(testing_data)
    conditional_decorators = _collect_conditional_decorators(testing_data)

    for skip_name in testing_data.skips:
        test_funcs[skip_name].add_mark("skip")

    for xfail_name in testing_data.xfails:
        test_funcs[xfail_name].add_mark("xfail")

    for func_name, tolerances in testing_data.tolerances.items():
        test_funcs[func_name].tolerances.update(tolerances)

    for test_func in test_funcs.values():
        attrs[test_func.name] = test_func.collect(
            testing_data,
            decorators=decorators,
            conditional_decorators=conditional_decorators,
            skip_vec=testing_data.skip_vec,
        )

    return attrs


class Parametrizer(type):
    """Metaclass for test classes driven by test definition.

    Note: A test class is a class that inherits from TestCase.
    For example, `class TestEuclidean(TestCase)` defines
    a test class.

    The Parametrizer helps its test class by pairing:
    - the different test functions of the test class:
      - e.g. the test function `test_belongs`,
    - with different test data, generated by auxiliary test data functions
      - e.g. the test data function `belongs_data` that generates data
      to test the function `belongs`.

    As such, Parametrizer acts as a "metaclass" of its test class:
    `class TestEuclidean(TestCase, metaclass=Parametrizer)`.

    Specifically, Parametrizer decorates every test function inside
    its test class with pytest.mark.parametrizer, with the exception
    of the test class' class methods and static methods.

    Two conventions need to be respected:
    1. The test class should contain an attribute named 'testing_data'.
      - `testing_data` is an object inheriting from `TestData`.
    2. Every test function should have its corresponding test data function created
    inside the TestData object called `testing_data`.

    A sample test class looks like this:

    ```
    class TestDataEuclidean(TestData):
        def belongs_data():
            ...
            return self.generate_tests(...)

    class TestEuclidean(TestCase, metaclass=Parametrizer):
        testing_data = TestDataEuclidean()
        def test_belongs():
            ...
    ```
    Parameters
    ----------
    cls : child class of TestCase
        Test class, i.e. a class inheriting from TestCase
    name : str
        Name of the test class
    bases : TestCase
        Parent class of the test class: TestCase.
    attrs : dict
        Attributes of the test class, for example its methods,
        stored in a dictionnary as (key, value) when key gives the
        name of the attribute (for example the name of the method),
        and value gives the actual attribute (for example the method
        itself.)

    References
    ----------
    More on pytest's parametrizers can be found here:
    https://docs.pytest.org/en/6.2.x/parametrize.html
    """

    def __new__(cls, name, bases, attrs):
        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        test_funcs = _collect_all_tests(attrs, bases, active=True)
        _check_test_data_pairing(test_funcs, testing_data)

        if testing_data.skip_all:
            for test_func in test_funcs.values():
                test_func.add_mark("skip")

        _update_attrs(test_funcs, testing_data, attrs)
        return super().__new__(cls, name, bases, attrs)


class DataBasedParametrizer(type):
    """Metaclass for test classes driven by data definition.

    It differs from `Parametrizer` because every test data function must have
    an associated test function, instead of the opposite.
    """

    def __new__(cls, name, bases, attrs):
        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        test_funcs = _collect_all_tests(attrs, bases, active=False)

        _activate_tests_given_data(test_funcs, testing_data)

        if testing_data.skip_all:
            for test_func in test_funcs.values():
                test_func.add_mark("skip")

        _update_attrs(test_funcs, testing_data, attrs)
        return super().__new__(cls, name, bases, attrs)


def _get_available_data_names(testing_data):
    return [
        _test_data_name_to_vanilla_name(attr_name)
        for attr_name in dir(testing_data)
        if _is_test_data(attr_name)
    ]


def _raise_missing_testing_data(testing_data):
    if testing_data is None:
        raise Exception("Testing class doesn't have class object named 'testing_data'")


def _collect_decorators(testing_data):
    decorators = []
    if not testing_data.fail_for_autodiff_exceptions and not autodiff_backend():
        decorators.append(_except_autodiff_exception)

    if not testing_data.fail_for_not_implemented_errors:
        decorators.append(_except_not_implemented_errors)

    return decorators


def _trial_condition(test_func):
    marks = test_func.marks
    return "vec" in marks or "random" in marks


def _collect_conditional_decorators(testing_data):
    decorators = []
    if testing_data.trials > 1:
        decorators.append(
            (
                _trial_condition,
                lambda fn: _multiple_trials(fn, trials=testing_data.trials),
            )
        )

    return decorators


def _get_pytest_mark_names(test_func):
    return [elem.name for elem in test_func.pytestmark if isinstance(elem, pytest.Mark)]


def _except_autodiff_exception(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutodiffNotImplementedError:
            pass

    return _wrapped


def _except_not_implemented_errors(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            pass

    return _wrapped


def _multiple_trials(func, trials):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        trial = 0
        while True:
            try:
                trial += 1
                return func(*args, **kwargs)
            except Exception as exception:
                if trial < trials:
                    continue

                raise exception

    return _wrapped


def _copy_func(
    f,
    name=None,
):
    """Copy function.

    Return a function with same code, globals, defaults, closure, and
    name (or provide a new name).

    Additionally, keyword arguments are transformed into positional arguments for
    compatibility with pytest.
    """
    fn = types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )
    fn.__dict__.update(f.__dict__)

    sign = inspect.signature(fn)
    defaults, new_params = {}, []
    for param in sign.parameters.values():
        if param.default is inspect._empty:
            new_params.append(param)
        else:
            new_params.append(inspect.Parameter(param.name, kind=1))
            defaults[param.name] = param.default
    new_sign = sign.replace(parameters=new_params)
    fn.__signature__ = new_sign

    return fn, defaults


def _is_test(attr_name):
    return attr_name.startswith("test_")


def _is_test_data(attr_name):
    return attr_name.endswith("_test_data")


def _test_data_name_to_vanilla_name(test_data_name):
    return test_data_name[:-10]


def _collect_available_tests(attrs):
    return {attr_name: attr for attr_name, attr in attrs.items() if _is_test(attr_name)}


def _collect_available_base_tests(bases):
    base_tests = dict()
    for base in bases:
        base_test_names = [name for name in dir(base) if _is_test(name)]
        for test_name in base_test_names:
            if test_name not in base_tests:
                base_tests[test_name] = getattr(base, test_name)

    return base_tests


def _collect_all_tests(attrs, bases, active=True):
    test_attrs = _collect_available_tests(attrs)
    base_attrs = _collect_available_base_tests(bases)

    # order matters
    tests = {**base_attrs, **test_attrs}

    all_tests = {}
    for name, func in tests.items():
        test_function = TestFunction(name, func, active=active)
        all_tests[test_function.vanilla_name] = test_function

    return all_tests


def _raise_missing_tests(missing_tests):
    if missing_tests:
        msg = "Need to define tests for:"
        for name in set(missing_tests):
            msg += f"\n\t-{name}_test_data"

        raise Exception(msg)


def _activate_tests_given_data(test_funcs, testing_data):
    defined_names_ls = _get_available_data_names(testing_data)

    missing_tests = []
    for name in defined_names_ls:
        if name in test_funcs:
            test_funcs[name].active = True

        elif not name.endswith("_vec"):
            missing_tests.append(name)
            continue

        else:
            name_no_vec = name[:-4]
            if name_no_vec not in test_funcs:
                missing_tests.append(name)
                continue

            test_function = AutoVecTestFunction(test_funcs[name_no_vec])
            test_funcs[test_function.vanilla_name] = test_function

    _raise_missing_tests(missing_tests)

    return test_funcs


def _raise_missing_data(missing_data):
    if missing_data:
        msg = "Need to define data for:"
        for name in set(missing_data):
            msg += f"\n\t-test_{name}"

        raise Exception(msg)


def _check_test_data_pairing(test_funcs, testing_data):
    defined_names_ls = _get_available_data_names(testing_data)
    missing_data = []
    for name in test_funcs.keys():
        if name not in defined_names_ls:
            missing_data.append(name)

    _raise_missing_data(missing_data)

    return test_funcs


def _run_example(path, **kwargs):
    warnings.simplefilter("ignore", category=UserWarning)

    spec = importlib.util.spec_from_file_location("module.name", path)
    example = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(example)

    example.main(**kwargs)
    plt.close("all")


class ExamplesParametrizer(type):
    def __new__(cls, name, bases, attrs):
        def _create_new_test(path, **kwargs):
            def new_test(self):
                return _run_example(path=path, **kwargs)

            return new_test

        BACKEND = os.environ.get("GEOMSTATS_BACKEND", "numpy")

        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        paths = testing_data.paths
        metadata = testing_data.metadata
        skips = testing_data.skips

        for path in paths:
            name = path.split(os.sep)[-1].split(".")[0]

            func_name = f"test_{name}"

            metadata_ = metadata.get(name, {})
            if not isinstance(metadata_, dict):
                metadata_ = {"backends": metadata_}

            kwargs = metadata_.get("kwargs", {})
            test_func = _create_new_test(path, **kwargs)

            backends = metadata_.get("backends", None)
            if name in skips or (backends and BACKEND not in backends):
                test_func = pytest.mark.skip()(test_func)

            attrs[func_name] = test_func

        return super().__new__(cls, name, bases, attrs)


def _exec_notebook(path):
    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    shutil.copy(path, file_name)

    with open(file_name) as file:
        notebook = nbformat.read(file, as_version=4)

    eprocessor = ExecutePreprocessor(timeout=1000, kernel_name="python3")
    eprocessor.preprocess(notebook)


def _has_package(package_name):
    """Check if package is installed.

    Parameters
    ----------
    package_name : str
        Package name.
    """
    return importlib.util.find_spec(package_name) is not None


class NotebooksParametrizer(type):
    def __new__(cls, name, bases, attrs):
        def _create_new_test(path, **kwargs):
            def new_test(self):
                return _exec_notebook(path=path)

            return new_test

        BACKEND = os.environ.get("GEOMSTATS_BACKEND", "numpy")

        testing_data = locals()["attrs"].get("testing_data")
        _raise_missing_testing_data(testing_data)

        paths = testing_data.paths

        for path in paths:
            name = path.split(os.sep)[-1].split(".")[0]

            func_name = f"test_{name}"
            test_func = _create_new_test(path)

            with open(path, "r", encoding="utf8") as file:
                metadata = json.load(file).get("metadata")

            backends = metadata.get("backends", None)
            requires = metadata.get("requires", [])
            if (backends and BACKEND not in backends) or not all(
                [_has_package(package) for package in requires]
            ):
                test_func = pytest.mark.skip()(test_func)

            attrs[func_name] = test_func

        return super().__new__(cls, name, bases, attrs)
