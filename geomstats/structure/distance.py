"""
TODO: Make generic object for Distance
"""
import abc

class Distance(abc.ABC):

    def __init__(self, space):
        self._space = space
