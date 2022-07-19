import logging
import random

from geomstats.learning.frechet_mean import FrechetMean


class AACFrechet:
    def __init__(
        self,
        metric,
        epsilon=1e-6,
        max_iter=20,
        init_point=None,
        mean_estimator_kwargs=None,
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        mean_estimator_kwargs = mean_estimator_kwargs or {}
        self.mean_estimator = FrechetMean(
            self.metric.total_space_metric, **mean_estimator_kwargs
        )

        self.estimate_ = None
        self.n_iter_ = None

    def fit(self, X):
        previous_estimate = (
            random.choice(X) if self.init_point is None else self.init_point
        )
        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1

            aligned_X = self.metric.align_point_to_point(previous_estimate, X)
            new_estimate = self.mean_estimator.fit(aligned_X).estimate_
            error = self.metric.total_space_metric.dist(previous_estimate, new_estimate)

            previous_estimate = new_estimate

        if iteration == self.max_iter:
            logging.warning(
                f"Maximum number of iterations {self.max_iter} reached. "
                "The mean may be inaccurate"
            )

        self.estimate_ = new_estimate
        self.n_iter_ = iteration

        return self
