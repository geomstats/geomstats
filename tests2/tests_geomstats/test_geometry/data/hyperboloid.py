from tests2.data.base_data import LevelSetTestData


class HyperboloidTestData(LevelSetTestData):
    tolerances = {"projection_belongs": {"atol": 1e-8}}
