import unittest
from sklearn.utils.estimator_checks import check_estimator

from geomstats.tests import TestCase
from geomstats.learning._template import (TemplateEstimator,
                                          TemplateTransformer,
                                          TemplateClassifier)


ESTIMATORS = (TemplateEstimator, TemplateTransformer, TemplateClassifier)


class TestEstimators(TestCase):
    def test_template_estimator(self):
        check_estimator(TemplateEstimator)

    def test_template_transformer(self):
        check_estimator(TemplateTransformer)

    def test_template_classifier(self):
        check_estimator(TemplateClassifier)
