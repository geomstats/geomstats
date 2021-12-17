# def metric(q):
#     # 3D Riemannian metric, i.e. a 3x3 matrix
#     # q : basepoint in M
    
#     return gs.array([
#         [1 + q[0]**2 , 1, 1],
#         [0, 1 + q[1]**2, 2],
#         [0,0, 1]])

# def hamiltonian_Riemannian(x2, metric):
#     """ 
#     metric: 3x3 matrix-valued function of a 3-array.
#     """

#     q, p = x
#     cometric = gs.linalg.inv(metric(q))
#     return 1/2 * gs.einsum('i,ij,j', p, cometric, p)

# def force(state, t):

#     x, v = state

#     return gs.array([x, v])

# def symplectic_euler_step(force, state, time, dt):
#     """Compute one step of the symplectic euler approximation.

#     Symplectic euler_integration of an equation system assumed to be of
#     the following form: we interpret the state variable as an element
#     of a fiber bundle, state = (x,v), where x is interpreted as a
#     position variable, and v is an element of the fiber (e.g. the tangent
#     or cotangent space at x). The equation system is the following,

#         d state / dt = force(t, state).

#     The force function is assumed to be of the form

#         force(t, state) = (f(t,v), g(t,x))

#     so the equation system can be written as

#         dx/dt = f(t, v)
#         dv/dt = g(t, x).

#     Parameters
#     ----------
#     state : array-like, shape=[2, dim]
#         State at time t, corresponds to position and velocity variables at
#         time t.
#     force : callable
#         Vector field that is being integrated.
#     dt : float
#         Time-step in the integration.

#     Returns
#     -------
#     point_new : array-like, shape=[,,,, {dim, [n, n]}]
#         First variable at time t + dt.
#     vector_new : array-like, shape=[,,,, {dim, [n, n]}]
#         Second variable at time t + dt.

#     References
#     ----------
#     https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
#     """
#     base_coordinate, fiber_coordinate = state
#     _, fiber_derivative = force(state, time)

#     new_fiber_coordinate = fiber_coordinate + fiber_derivative * dt

#     sympl_base_derivative, _ = force(gs.array([base_coordinate,
#                                                new_fiber_coordinate]),
#                                      time)

#     new_base_coordinate = base_coordinate + sympl_base_derivative * dt

#     new_state = gs.array([new_base_coordinate, new_fiber_coordinate])

#     return new_state


# state = gs.array([[1,1,1.],[4.,4,4]])
# time = 2
# dt = 0.01

# symplectic_euler_step(force, state, time, dt)



# _______________________________________


####### instantiate a SR metric class

import geomstats.backend as gs
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric


class exampleMetric(SubRiemannianMetric):
    """Class for Sub-Riemannian metrics.

    This implementation assumes a distribution of constant dimension.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    dist_dim : int
        Dimension of the distribution
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, dist_dim, default_point_type="vector"):
        super(exampleMetric, self).__init__(
            dim=dim, dist_dim=dist_dim, default_point_type=default_point_type
        )

    def cometric_matrix(self, base_point=None):
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inverse of inner-product matrix.
        """

        return gs.array([[base_point[0], 0., 0.],
                         [0., base_point[1], 0.],
                         [0., 0., base_point[2]]])

cotangent_vec = gs.array([1., 1., 1.])
base_point = gs.array([2., 1., 10.])
N_STEPS = 20

e = exampleMetric(dim=3, dist_dim=2)

e.exp(cotangent_vec, base_point, n_steps=N_STEPS)
