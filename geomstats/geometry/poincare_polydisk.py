"""
The Poincare polydisk
"""

from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.geometry.product_manifold import ProductManifold


class PoincarePolydisk(ProductManifold):
    """
    Class for the Poincare polydisk, which is a direct product
    of n Poincare disks, i.e. hyperbolic spaces of dimension 2.
    """
    def __init__(self, n_disks):
        disk = HyperbolicSpace(dimension=2)
        list_disks = [disk, ] * n_disks
        super(PoincarePolydisk, self).__init__(
            manifolds=list_disks)
