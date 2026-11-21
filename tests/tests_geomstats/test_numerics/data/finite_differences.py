import random

from polpo.testing.geometric import GeometricCaseData


class ForwardDifferenceTestData(GeometricCaseData):
    def forward_difference_last_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            n_times=random.randint(5, 9),
        )


class CenteredDifferenceTestData(GeometricCaseData):
    def centered_difference_random_index_test_data(self):
        n_times = random.randint(5, 9)
        return [
            datum
            for endpoints in (True, False)
            for datum in self.generate_random_data(
                arg_names=("base_point", "tangent_vec"),
                n_times=n_times,
                endpoints=endpoints,
            )
        ]


class SecondCenteredDifferenceTestData(GeometricCaseData):
    def second_centered_difference_random_index_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            n_times=random.randint(5, 9),
        )
