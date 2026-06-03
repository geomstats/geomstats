"""Learning algorithms on manifolds."""

__all__ = ["TemplateEstimator", "TemplateClassifier", "TemplateTransformer"]

from ._sklearn import _enable_array_dispatch
from ._template import TemplateClassifier, TemplateEstimator, TemplateTransformer

_enable_array_dispatch()
