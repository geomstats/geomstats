import random

from polpo.testing.lazy import LazyValue


class InformationManifoldMixinTestData:
    _N_SAMPLES = [1] + random.sample(range(2, 5), 1)

    def _random_support_points(self, n_samples):
        return self.space.sample(self.space.random_point(), n_samples)

    def _generate_pdf_vectorization_data(self):
        data = []
        for n_samples in self._N_SAMPLES:
            x = LazyValue(
                self._random_support_points, n_samples, label=f"x={n_samples}"
            )
            data.extend(
                self.generate_vectorization_data(
                    arg_names="point",
                    op_name="point_to_pdf",
                    op_evaluator=lambda op, point, x=x: op(point)(x.materialize()),
                    x=x,
                )
            )
        return data

    def _generate_random_data_with_samples(self):
        data = []
        for n_samples in self._N_SAMPLES:
            data.extend(
                self.generate_random_data(arg_names="point", n_samples=n_samples)
            )
        return data

    def point_to_pdf_vec_test_data(self):
        return self._generate_pdf_vectorization_data()

    def sample_shape_test_data(self):
        return self._generate_random_data_with_samples()

    def sample_belongs_to_support_test_data(self):
        return self._generate_random_data_with_samples()
