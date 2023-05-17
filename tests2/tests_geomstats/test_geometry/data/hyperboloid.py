from .base import LevelSetTestData


class HyperboloidTestData(LevelSetTestData):
    tolerances = {"projection_belongs": {"atol": 1e-8}}
