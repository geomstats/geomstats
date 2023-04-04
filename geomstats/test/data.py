import random


class TestData:
    """Class for TestData objects."""

    N_VEC_REPS = random.sample(range(2, 5), 1)
    N_SHAPE_POINTS = [1] + random.sample(range(2, 5), 1)
    N_RANDOM_POINTS = [1] + random.sample(range(2, 5), 1)
    N_TIME_POINTS = [1] + random.sample(range(2, 5), 1)

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

    def generate_random_data(self, marks=()):
        data = [dict(n_points=n_points) for n_points in self.N_RANDOM_POINTS]
        return self.generate_tests(data, marks=marks)

    def generate_vec_data(self, marks=()):
        data = [dict(n_reps=n_reps) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data, marks=marks)

    def generate_shape_data(self, marks=()):
        data = [dict(n_points=n_points) for n_points in self.N_SHAPE_POINTS]
        return self.generate_tests(data, marks=marks)

    def generate_vec_data_with_time(self, marks=()):
        data = []
        for n_reps in self.N_VEC_REPS:
            for n_times in self.N_TIME_POINTS:
                data.append(dict(n_reps=n_reps, n_times=n_times))

        return self.generate_tests(data, marks=marks)
