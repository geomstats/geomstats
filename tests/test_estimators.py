from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator

import geomstats.backend as gs
assert_allclose = gs.testing.assert_allclose
from geomstats.tests import TestCase
from geomstats.learning._template import (TemplateEstimator,
                                          TemplateTransformer,
                                          TemplateClassifier)


ESTIMATORS = (TemplateEstimator, TemplateTransformer, TemplateClassifier)

# TODO(nkoep): Rewrite bare assertS to use self.assert*

# XXX: Should these tests run on all backends?

class TestEstimators(TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.data = load_iris(return_X_y=True)

    def test_check_template_estimator(self):
        check_estimator(TemplateEstimator)

    def test_check_template_transformer(self):
        check_estimator(TemplateTransformer)

    def test_check_template_classifier(self):
        check_estimator(TemplateClassifier)

    def test_template_estimator(self):
        est = TemplateEstimator()
        self.assertEqual(est.demo_param, 'demo_param')

        X, y = self.data

        est.fit(X, y)
        self.assertTrue(hasattr(est, 'is_fitted_'))

        y_pred = est.predict(X)
        assert_allclose(y_pred, gs.ones(X.shape[0], dtype=gs.int64))

    def test_template_transformer_error(self):
        X, y = self.data
        trans = TemplateTransformer()
        trans.fit(X)
        X_diff_size = np.ones((10, X.shape[1] + 1))
        self.assertRaises(trans.transform(X_diff_size), ValueError)

    def test_template_transformer(self):
        X, y = self.data
        trans = TemplateTransformer()
        assert trans.demo_param == 'demo'

        trans.fit(X)
        assert trans.n_features_ == X.shape[1]

        X_trans = trans.transform(X)
        assert_allclose(X_trans, np.sqrt(X))

        X_trans = trans.fit_transform(X)
        assert_allclose(X_trans, np.sqrt(X))

    def test_template_classifier(self):
        X, y = self.data
        clf = TemplateClassifier()
        assert clf.demo_param == 'demo'

        clf.fit(X, y)
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'X_')
        assert hasattr(clf, 'y_')

        y_pred = clf.predict(X)
        assert y_pred.shape == (X.shape[0],)
