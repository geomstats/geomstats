"""Abstract class for groups."""

import abc


class Group(abc.ABC):

    @abc.abstractmethod
    def identity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compose(self, g1, g2):
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self):
        raise NotImplementedError

    @abc.abstractmethod
    def irrep(self, index):
        # TODO idea for later
            # def irrep(self, index):
            #     return lambda x: gs.exp(1j * theta x)
        raise NotImplementedError

    def left_translation(self, g):
        return lambda h: self.compose(g, h)

    def right_translation(self, g):
        return lambda h: self.compose(h, g)

    def conjugation(self, g, h):
        return lambda h: self.compose(self.compose(g, h), self.inverse(g))
