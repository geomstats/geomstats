"""Abstract class for Lie groups."""

from geomstats.spaces.core import Manifold, Group
import geomstats.backend as gs
from geomstats import matrices


class LieGroup(Group, Manifold):

    def __init__(self, dim, lie_algebra=None, **kwargs):
        self.lie_algebra = lie_algebra

    def jacobian_left_translation(self, g, at):
        raise NotImplementedError

    def jacobian_right_translation(self, g, at):
        raise NotImplementedError

    def jacobian_conjugation(self, g, at):
        raise NotImplementedError

    def diff_left_translation(self, g, at):
        return lambda u: matrices.mul(self.jacobian_left_translation(g, at), u)

    def diff_right_translation(self, g, at):
        return lambda u: matrices.mul(self.jacobian_right_translation(g, at), u)

    def diff_conjugation(self, g, at):
        return lambda u: matrices.mul(self.jacobian_conjugation(g, at), u)

    def adjoint_representation(self, g):
        return self.jacobian_conjugation(g, self.identity)

    def jacobian_adjoint_representation(self, at):
        raise NotImplementationError

    def diff_adjoint_representation(self, at):
        raise lambda u: matrices.mul(self.jacobian_adjoint_representation(at), u)

    def lie_bracket(self, u, v, base_g=None):
        if base_g is None:
            return self.lie_algebra.bracket(u, v)
        inverse_base_g = self.inverse(base_g)

        u_at_identity = self.diff_left_translation(inverse_base_g, base_g)(u)
        v_at_identity = self.diff_left_translation(inverse_base_g, base_g)(v)

        bracket_at_identity = self.lie_algebra.bracket(u_at_identity, v_at_identity)
        bracket = self.diff_left_translation(base_g, self.identity)(bracket_at_identity)
        return bracket
