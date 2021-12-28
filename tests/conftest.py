import inspect
import types

import pytest

smoke = pytest.mark.smoke
random = pytest.mark.rt


def generate_tests(smoke_test_data, random_test_data=[]):
    smoke_tests = [
        pytest.param(*data.values(), marks=smoke) for data in smoke_test_data
    ]
    random_tests = [pytest.param(*data, marks=random) for data in random_test_data]
    return smoke_tests + random_tests


class Parametrizer(type):
    def __new__(cls, name, bases, attrs):
        print(locals())
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):
                print("attr_name", attr_name)
                print("attr_value", attr_value)
                if attr_name.startswith("test_"):
                    args_str = ", ".join(inspect.getfullargspec(attr_value)[0][1:])
                    data_fn_str = attr_name[5:] + "_data"
                    attrs[attr_name] = pytest.mark.parametrize(
                        args_str, locals()[data_fn_str]()
                    )(attr_value)

        return super(Parametrizer, cls).__new__(cls, name, bases, attrs)
