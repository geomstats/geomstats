from time import time

from autograd import elementwise_grad, value_and_grad
from autograd.scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.optimize import minimize
from spd_tools.divergences_mpe import mixed_power_euclidean_cometric as cometric
from spd_tools.divergences_mpe import mixed_power_euclidean_flat as flat

import geomstats.backend as gs


class MPEGeodesics(object):
    def __init__(self, power1=1.0, power2=2.0, n_steps=10, scheme="euler"):
        self.power1 = power1
        self.power2 = power2
        self.n_steps = n_steps
        self.scheme = scheme

    def exp(self, tangent_vec, base_point):
        tangent_vec = self.make_symmetric(tangent_vec)
        momentum = flat(self.power1, self.power2, tangent_vec, base_point)
        state = gs.stack([base_point, momentum])
        flow = self.symp_flow(
            self.kinetic_energy, n_steps=self.n_steps, scheme=self.scheme
        )(state)
        return flow[-1][0]

    def scipy_exp(self, tangent_vec, base_point):
        momentum = flat(self.power1, self.power2, tangent_vec, base_point)

        def ivp(flat_state, t):
            state_ = flat_state.reshape(gs.stack([base_point, momentum]).shape)
            return self.symp_grad(self.kinetic_energy)(state_).flatten()

        times = gs.linspace(0, 1, self.n_steps + 1)
        initial_state = gs.hstack([base_point, momentum]).flatten()
        geodesic = odeint(ivp, initial_state, times, tuple(), rtol=1e-6)
        return geodesic[-1].reshape(gs.stack([base_point, momentum]).shape)[0]

    def log(self, point, base_point):
        point_ = gs.copy(point)
        if point_.ndim == 2:
            point_ = gs.stack([point_])
        if base_point.ndim == 2:
            base_point = gs.stack([base_point])
        s = time()

        def objective(velocity):
            """Define the objective function."""
            velocity = velocity.reshape(point_.shape)
            delta = self.exp(velocity, base_point) - point_
            res = 1.0 / len(base_point) * gs.sum(delta ** 2)
            return res

        objective_with_grad = value_and_grad(objective)

        tangent_vec = gs.zeros_like(base_point).flatten()
        res = minimize(
            objective_with_grad,
            tangent_vec,
            method="L-BFGS-B",
            jac=True,
            options={"disp": True, "maxiter": 50},
            tol=1e-6,
        )
        print(f"Finished shooting in {time() - s} sec")
        tangent_vec = self.make_symmetric(gs.reshape(res.x, base_point.shape))
        return tangent_vec

    def scipy_log(self, point, base_point):
        stop_time = 1.0
        t = gs.linspace(0, stop_time, self.n_steps + 2)
        s = time()

        def bvp(time, flat_states):
            result = []
            for flat_state in flat_states.T:
                state_ = flat_state.reshape(gs.stack([base_point, point]).shape)
                result.append(self.symp_grad(self.kinetic_energy)(state_).flatten())
            return gs.stack(result).T

        def boundary_cond(flat_state_0, flat_state_1):
            state_0 = flat_state_0.reshape((2, -1))
            state_1 = flat_state_1.reshape((2, -1))
            return gs.array(
                [state_0[0] - base_point.flatten(), state_1[0] - point.flatten()]
            ).flatten()

        geodesic_init = gs.einsum("n,kij->nkij", t, point) + gs.einsum(
            "n,kij->nkij", 1 - t, base_point
        )
        velocity_init = gs.stack([gs.zeros_like(base_point)] * (self.n_steps + 2))
        state_init = gs.stack([geodesic_init, velocity_init], axis=1)
        state_init = state_init.transpose(1, 2, 3, 4, 0).reshape((-1, self.n_steps + 2))
        solution = solve_bvp(bvp, boundary_cond, t, state_init, tol=1e-6)
        geodesic = solution.y
        geodesic = geodesic[:, :2]
        velocity = 1.0 / solution.x[1] * (geodesic[:, 1] - geodesic[:, 0])
        velocity = velocity.reshape((2, -1))[0].reshape(base_point.shape)
        print(f"Finished path straightening in {time() - s} sec")

        return velocity

    def kinetic_energy(self, state):
        position, momentum = state
        momentum = self.make_symmetric(momentum)
        # position = self.make_symmetric(position)
        return 1 / 2 * cometric(self.power1, self.power2, momentum, momentum, position)

    @staticmethod
    def make_symmetric(mat):
        return 1.0 / 2 * (mat + gs.transpose(mat, axes=(0, 2, 1)))

    @classmethod
    def symp_flow(cls, hamiltonian, end_time=1.0, n_steps=20, scheme="euler"):
        if scheme == "rk2":
            step = cls.rk2
        elif scheme == "leapfrog":
            step = cls.leapfrog
        elif scheme == "rk4":
            step = cls.rk4
        elif scheme == "yoshida":
            step = cls.yoshida
        else:
            step = cls.symp_euler
        step_size = end_time / n_steps
        return cls.iterate(step(hamiltonian, step_size), n_steps)

    @classmethod
    def symp_grad(cls, hamiltonian):
        def vector(x):
            H_q, H_p = elementwise_grad(hamiltonian)(x)
            return gs.array([H_p, -H_q])

        return vector

    @classmethod
    def symp_euler(cls, hamiltonian, step_size):
        def step(state):
            position, momentum = state
            dq, _ = cls.symp_grad(hamiltonian)(state)
            y = gs.array([position + dq, momentum])
            _, dp = cls.symp_grad(hamiltonian)(y)
            return gs.array([position + step_size * dq, momentum + step_size * dp])

        return step

    @classmethod
    def rk2(cls, hamiltonian, step_size):
        def step(state):
            dt = step_size
            position, momentum = state
            k1, l1 = cls.symp_grad(hamiltonian)(state)
            y1 = gs.stack([position + dt / 2 * k1, momentum + dt / 2 * l1])

            k2, l2 = cls.symp_grad(hamiltonian)(y1)
            return position + dt * k2, momentum + dt * l2

        return step

    @classmethod
    def leapfrog(cls, hamiltonian, step_size):
        def step(state):
            dt = step_size
            position, momentum = state
            _, l1 = cls.symp_grad(hamiltonian)(state)
            y1 = gs.stack([position, momentum + dt / 2 * l1])

            k2, _ = cls.symp_grad(hamiltonian)(y1)
            y2 = gs.stack([position + dt * k2, y1[1]])

            _, l2 = cls.symp_grad(hamiltonian)(y2)

            return y2[0], y1[1] + dt / 2 * l2

        return step

    @classmethod
    def rk4(cls, hamiltonian, step_size):
        def step(state):
            dt = step_size
            position, momentum = state
            k1, l1 = cls.symp_grad(hamiltonian)(state)
            y1 = gs.stack([position + dt / 2 * k1, momentum + dt / 2 * l1])

            k2, l2 = cls.symp_grad(hamiltonian)(y1)
            y2 = gs.stack([position + dt / 2 * k2, momentum + dt / 2 * l2])

            k3, l3 = cls.symp_grad(hamiltonian)(y2)
            y3 = gs.stack([position + dt * k3, momentum + dt * l3])

            k4, l4 = cls.symp_grad(hamiltonian)(y3)
            position_new = position + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            momentum_new = momentum + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
            return position_new, momentum_new

        return step

    @classmethod
    def yoshida(cls, hamiltonian, step_size):
        c1 = 1 / (2 * (2 - 2 ** (1.0 / 3)))
        c2 = 1.0 / 2 - c1
        d1 = 2 * c1
        d2 = -(2 ** (1.0 / 3)) / (2 - 2 ** (1.0 / 3))

        def step(state):
            dt = step_size
            position, momentum = state
            k1, _ = cls.symp_grad(hamiltonian)(state)
            y1 = gs.stack([position + c1 * dt * k1, momentum])

            _, l1 = cls.symp_grad(hamiltonian)(y1)
            y2 = gs.stack([y1[0], momentum + d1 * dt * l1])

            k2, _ = cls.symp_grad(hamiltonian)(y2)
            y3 = gs.stack([y1[0] + c2 * dt * k2, y1[1]])

            _, l2 = cls.symp_grad(hamiltonian)(y3)
            y4 = gs.stack([y3[0], y2[1] + d2 * dt * l2])

            k3, _ = cls.symp_grad(hamiltonian)(y4)
            y5 = gs.stack([y4[0] + c2 * dt * k3, y4[1]])

            _, l3 = cls.symp_grad(hamiltonian)(y5)
            y6 = gs.stack([y5[0], y4[1] + d1 * dt * l3])

            k4, _ = cls.symp_grad(hamiltonian)(y6)
            return y6[0] + c1 * dt * k4, y6[1]

        return step

    @staticmethod
    def iterate(func, n_steps):
        def flow(x):
            xs = [x]
            for i in range(n_steps):
                xs.append(func(xs[i]))
            return gs.array(xs)

        return flow
