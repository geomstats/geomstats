import pytest

smoke = pytest.mark.smoke
random = pytest.mark.rt


def generate_tests(smoke_test_data, random_test_data=[]):
    smoke_tests = [
        pytest.param(*data.values(), marks=smoke) for data in smoke_test_data
    ]
    random_tests = [pytest.param(*data, marks=random) for data in random_test_data]
    return smoke_tests + random_tests
