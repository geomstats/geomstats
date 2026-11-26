from polpo.testing.geometric import GeometricCaseData


class UniformlySampledPathEnergyTestData(GeometricCaseData):
    def dist_from_path_energy_per_time_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"), n_times=100)
