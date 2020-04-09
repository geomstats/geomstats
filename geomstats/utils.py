import geomstats.backend as gs


def from_vector_to_diagonal_matrix(x):
    n = gs.shape(x)[-1]
    identity_n = gs.eye(n)
    diagonals = gs.einsum('ki,ij->kij', x, identity_n)
    return diagonals
