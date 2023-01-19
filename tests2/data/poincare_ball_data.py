from tests2.data.base_data import OpenSetTestData


class PoincareBallTestData(OpenSetTestData):
    xfails = ("projection_belongs",)
