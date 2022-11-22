import pytest


class TestData:
    """Class for TestData objects."""

    def __init__(self, space):
        self.space = space

    def generate_tests(self, smoke_test_data, random_test_data=()):
        """Wrap test data with corresponding markers.

        Parameters
        ----------
        smoke_test_data : list
            Test data that will be marked as smoke.

        random_test_data : list
            Test data that will be marked as random.
            Optional, default: []

        Returns
        -------
        _: list
            Tests.
        """
        tests = []
        for test_data, marker in zip(
            [smoke_test_data, random_test_data], [pytest.mark.smoke, pytest.mark.random]
        ):
            for test_datum in test_data:
                if isinstance(test_datum, dict):
                    test_datum["marks"] = marker
                else:
                    test_datum = list(test_datum)
                    test_datum.append(marker)

                tests.append(test_datum)

        return tests
