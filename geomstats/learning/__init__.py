"""Learning algorithms on manifolds."""

__all__ = [
    "TemplateEstimator",
    "TemplateClassifier",
    "TemplateTransformer",
    "PrincipalNestedSpheres",
]

from ._template import TemplateClassifier, TemplateEstimator, TemplateTransformer
from .principal_nested_spheres import PrincipalNestedSpheres
