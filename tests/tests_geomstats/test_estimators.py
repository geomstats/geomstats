"""Template unit tests for scikit-learn estimators."""

import pytest
from sklearn.datasets import load_iris

import geomstats.backend as gs
import tests.conftest
from geomstats.learning._template import (
    TemplateClassifier,
    TemplateEstimator,
    TemplateTransformer,
)

ESTIMATORS = (TemplateClassifier, TemplateEstimator, TemplateTransformer)


class TestEstimators(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        self.data = load_iris(return_X_y=True)

    @tests.conftest.np_and_autograd_only
    def test_template_estimator(self):
        est = TemplateEstimator()
        self.assertEqual(est.demo_param, "demo_param")

        X, y = self.data

        est.fit(X, y)
        self.assertTrue(hasattr(est, "is_fitted_"))

        y_pred = est.predict(X)
        self.assertAllClose(y_pred, gs.ones(gs.shape(X)[0]))

    @tests.conftest.np_and_autograd_only
    def test_template_transformer_error(self):
        X, _ = self.data
        n_samples = gs.shape(X)[0]
        trans = TemplateTransformer()
        trans.fit(X)
        X_diff_size = gs.ones((n_samples, gs.shape(X)[1] + 1))
        with pytest.raises(ValueError):
            trans.transform(X_diff_size)

    def test_template_transformer(self):
        X, _ = self.data
        trans = TemplateTransformer()
        self.assertTrue(trans.demo_param == "demo")

        trans.fit(X)
        self.assertTrue(trans.n_features_ == X.shape[1])

        X_trans = trans.transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))

        X_trans = trans.fit_transform(X)
        self.assertAllClose(X_trans, gs.sqrt(X))

    @tests.conftest.np_autograd_and_tf_only
    def test_template_classifier(self):
        X, y = self.data
        clf = TemplateClassifier()
        self.assertTrue(clf.demo_param == "demo")

        clf.fit(X, y)
        self.assertTrue(hasattr(clf, "classes_"))
        self.assertTrue(hasattr(clf, "X_"))
        self.assertTrue(hasattr(clf, "y_"))

        y_pred = clf.predict(X)
        self.assertTrue(y_pred.shape == (X.shape[0],))
