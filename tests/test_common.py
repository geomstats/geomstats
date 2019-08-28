import pytest

from sklearn.utils.estimator_checks import check_estimator

from geomstats.learning._template import TemplateEstimator
from geomstats.learning._template import TemplateClassifier
from geomstats.learning._template import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
