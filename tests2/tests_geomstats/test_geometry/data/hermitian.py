from .base import ComplexVectorSpaceTestData
from .mixins import GroupExpMixinsTestData


class HermitianTestData(GroupExpMixinsTestData, ComplexVectorSpaceTestData):
    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])
