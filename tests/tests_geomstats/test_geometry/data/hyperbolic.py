from polpo.testing.data import CaseData

import geomstats.backend as gs


class HalfSpaceToBall2TestData(CaseData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.0, 1.0]),
                expected=gs.zeros(2),
            ),
            dict(
                base_point=gs.array([[0.0, 1.0], [0.0, 2.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 1.0 / 3.0]]),
            ),
        ]
        return data


class ExtrinsicToBall3TestData(CaseData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([2.0, 1.0, gs.sqrt(2)]),
                expected=gs.array([1.0 / 3.0, gs.sqrt(2) / 3.0]),
            ),
        ]
        return data

    def tangent_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([1.0, 0.0, 0.0]),
                expected=gs.array([1.0 / 2.0, 1.0 / 2.0]),
            )
        ]
        return data
