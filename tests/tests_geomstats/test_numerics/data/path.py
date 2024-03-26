from geomstats.test.data import TestData


class UniformlySampledPathEnergyTestData(TestData):
    N_TIME_POINTS = [100]

    tolerances = {"dist_from_path_energy_per_time": {"atol": 1e-2}}

    def dist_from_path_energy_per_time_test_data(self):
        return self.generate_random_data_with_time()
