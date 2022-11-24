class TestData:
    """Class for TestData objects."""

    def generate_tests(self, test_data, marks=()):
        """Wrap test data with corresponding marks.

        Parameters
        ----------
        test_data : list or dict
        marks : list
            pytest marks,

        Returns
        -------
        data: list or dict
            Tests.
        """

        tests = []
        if not isinstance(marks, (list, tuple)):
            marks = [marks]

        for test_datum in test_data:

            if isinstance(test_datum, dict):
                if "marks" not in test_datum:
                    test_datum["marks"] = marks
                else:
                    test_datum["marks"].extend(marks)

            else:
                test_datum = list(test_datum)
                test_datum.append(marks)

            tests.append(test_datum)

        return tests
