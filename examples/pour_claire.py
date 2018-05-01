"""
"""

import matplotlib.pyplot as plt
import numpy as np

import spd_matrices_space as spd_space
from geomstats.spd_matrices_space import SPDMatricesSpaces

SPACE = SPDMatricesSpace(n=...)


def main():
    hat_l1 = ...
    hat_l2 = ...
    log_euclidean_distance = Frobenius(spd_space.group_log(point=hat_l1) - spd_space.group_log(point=hat_l2))
    inv_sqrt_hat_l1 = np.linalg.inv(scipy.linalg.sqrtm(hat_l1))
    riemannian_distance = Frobenius(spd_space.group_log(inv_sqrt_hat_l1 @ hat_l2 @ inv_sqrt_hat_l1)


if __name__ == "__main__":
    main()
