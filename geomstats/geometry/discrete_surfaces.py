"""Discrete Surfaces with Elastic metrics.
Lead author: Emmanuel Hartman
"""
import math

import numpy as np

##
from scipy.optimize import minimize, root

## Needs to be replaced with backend
from torch.autograd import grad

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class DiscreteSurfaces(Manifold):
    r"""Space of discrete surfaces sampled with nV vertices and nF faces in $\mathbb{R}^3$.

    Each individual surfaceis represented by a 2d-array of shape `[
    nV, 3]`. This space corresponds to the space of immersions defined below, i.e. the
    space of smooth functions from atemplate 2 manifold M into  $\mathbb{R}^3$,
    with non-vanishing Jacobian.
    .. math::
        Imm(M,\mathbb{R}^3)=\{ f \in C^{\infty}(M, \mathbb{R}^3) \|Df(x)\|\neq 0 \forall x \in M \}.

    Parameters
    ----------
    faces : integer array-like, shape=[nF, 3]

    Attributes
    ----------
    faces : integer array-like, shape=[nF, 3]
    """

    def __init__(self, faces, **kwargs):
        dim = (gs.amax(faces) + 1) * 3
        self.faces = faces

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold. Checks that vertices are inputed in proper form an are consistent with the mesh structure. Also checks if the discrete surface has degenerate triangles.

        Parameters
        ----------
        point :
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.
        Returns
        -------
        belongs : bool
            Boolean evaluating if point belongs to the space of discrete
            surfaces.
        """
        if point.shape[-1] != 3:
            return False
        if gs.amax(self.faces) + 1 != point.shape[-2]:
            return False
        alpha = self.getOneForms(point)
        g = gs.matmul(gs.transpose(alpha, (1, 2)), alpha)
        smallest_area = gs.min(self.get_face_areas(point))
        if smallest_area < atol:
            return False
        return True

    def is_tangent(self, vector, base_point, atol=gs.atol):
        return vector.shape[-1] == 3 and gs.amax(self.faces) + 1 == vector.shape[-2]

    def to_tangent(self, vector, base_point):
        raise NotImplementedError("to tangent is not implemented for discrete surfaces")

    def projection(self, point):
        raise NotImplementedError("projection is not implemented for discrete surfaces")

    def random_point(self, n_samples=1):
        """Producing random surfaces with vertices distibuted uniformly on the 2-sphere.
        Parameters
            ----------
            n_samples : int
                vertices of the triangulated surface
                Optional, Default=1
            Returns
            -------
            vertices :  array-like, shape=[n_samples,nV,3]
                the vertices for a batch of points in the space of discrete curves
        """
        Sphere = Hypersphere(2, "extrinsic")
        vertices = Sphere.random_uniform(n_samples * (gs.amax(self.faces) + 1))
        vertices = gs.reshape(vertices, (n_samples, gs.amax(self.faces) + 1, 3))
        print(vertices.shape)
        return vertices

    def get_vertex_areas(self, point):
        """Computation of vertex areas for a triangulated surface.
        Parameters
        ----------
        point : array-like, shape=[nVx3]
            vertices of the triangulated surface
        Returns
        -------
        vertareas :  array-like, shape=[nVx1]
            vertex areas
        """
        nV = point.shape[0]
        face_coordinates = point[self.faces]
        v0, v1, v2 = (
            face_coordinates[:, 0],
            face_coordinates[:, 1],
            face_coordinates[:, 2],
        )
        A = gs.linalg.norm((v1 - v2), axis=1)
        B = gs.linalg.norm((v0 - v2), axis=1)
        C = gs.linalg.norm((v0 - v1), axis=1)
        s = 0.5 * (A + B + C)
        area = gs.sqrt((s * (s - A) * (s - B) * (s - C)).clamp(min=1e-6))
        idx = gs.flatten(self.faces)
        incident_areas = gs.zeros(nV)
        val = gs.flatten(gs.stack([area] * 3, axis=1))
        incident_areas.scatter_add_(0, idx, val)
        vertAreas = 2 * incident_areas / 3.0
        return vertAreas

    def get_laplacian(self, point):
        """Computation of the mesh Laplacian operator of a triangulated surface evaluated at one of its tangent vectors h.
        Parameters
        ----------
        point  :  array-like, shape=[nVx3]
            vertices of the triangulated surface
        Returns
        -------
        L : function
            function that will evaluate the mesh Laplacian operator at a tangent vector to the surface
        """
        nV, nF = point.shape[0], self.faces.shape[0]
        face_coordinates = point[self.faces]
        v0, v1, v2 = (
            face_coordinates[:, 0],
            face_coordinates[:, 1],
            face_coordinates[:, 2],
        )
        A = gs.linalg.norm((v1 - v2), axis=1)
        B = gs.linalg.norm((v0 - v2), axis=1)
        C = gs.linalg.norm((v0 - v1), axis=1)
        s = 0.5 * (A + B + C)
        area = gs.sqrt((s * (s - A) * (s - B) * (s - C)).clamp(min=1e-6))
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = gs.stack([cota, cotb, cotc], axis=1)
        cot /= 2.0
        ii = self.faces[:, [1, 2, 0]]
        jj = self.faces[:, [2, 0, 1]]
        idx = gs.reshape(gs.stack([ii, jj], axis=0), (2, nF * 3))

        def L(h):
            """Function that evaluates the mesh Laplacian operator at a tangent vector to the surface.
            Parameters
            ----------
            h :  array-like, shape=[nVx3]
                tangent vector to the triangulated surface
            Returns
            -------
            Lh: array-like, shape=[nVx3]
                mesh Laplacian operator of the triangulated surface applied to one its tangent vector h
            """
            hdiff = h[idx[0]] - h[idx[1]]
            values = gs.stack([gs.flatten(cot)] * 3, axis=1) * hdiff
            Lh = gs.zeros((nV, 3))
            Lh[:, 0] = Lh[:, 0].scatter_add(0, idx[1, :], values[:, 0])
            Lh[:, 1] = Lh[:, 1].scatter_add(0, idx[1, :], values[:, 1])
            Lh[:, 2] = Lh[:, 2].scatter_add(0, idx[1, :], values[:, 2])
            return Lh

        return L

    def get_normal(self, point):
        """Computation of normals at each face of a triangulated surface.
        Parameters
        ----------
        point :  array-like, shape=[nVx3]
            vertices of the triangulated surface
        Returns
        -------
        N :  array-like, shape=[nFx1]
            normals of each face of the mesh
        """
        # Compute normals at each face by taking the cross product between edges of each face that are incident to its x-coordinate
        V0, V1, V2 = (
            point.index_select(0, self.faces[:, 0]),
            point.index_select(0, self.faces[:, 1]),
            point.index_select(0, self.faces[:, 2]),
        )
        N = 0.5 * gs.cross(V1 - V0, V2 - V0)
        return N

    def get_mesh_one_forms(self, point):
        """Computation of the vector valued one-forms evaluated at the faces of a triangulated surface.
        Parameters
        ----------
        point :  array-like, shape=[nVx3]
            vertices of the triangulated surface
        Returns
        -------
        alpha : array-like, shape=[nFx3x2]
            One form evaluated at each face of the triangulated surface
        """
        V0, V1, V2 = (
            point.index_select(0, self.faces[:, 0]),
            point.index_select(0, self.faces[:, 1]),
            point.index_select(0, self.faces[:, 2]),
        )
        return gs.stack([V1 - V0, V2 - V0], axis=2)

    def get_face_areas(self, point):
        """Computation of the areas for each face of a triangulated surface.
        Parameters
        ----------
        point :  array-like, shape=[nVx3]
            vertices of the triangulated surface
        Returns
        -------
        faceareas :  array-like, shape=[nFx2x2]
            Area computed at each face of the triangulated surface
        """
        g = self.get_surface_metric(point)
        return gs.sqrt(gs.linalg.det(g))

    def get_surface_metric(self, point):
        """Computation of the surface metric matrices evaluated at the faces of a triangulated surface.

        Parameters
        ----------
        point : array like, shape=[nVx3]
            vertices of the triangulated surface

        Returns
        -------
        g : array-like, shape=[nFx3x2]
            Surface metric matrix evaluated at each face of the triangulated surface
        """
        alpha = self.get_surface_one_forms(point)
        return gs.matmul(gs.transpose(alpha, (1, 2)), alpha)


class ElasticMetric(RiemannianMetric):
    def __init__(self, space, a0, a1, b1, c1, d1, a2):
        """Elastic metric defined a family of second order Sobolev metrics.
        Each individual surface is represented by a 2d-array of shape `[
        nV, 3]`.
        See [HSKCB2022]_ for details.
        Parameters
        ----------
        space : Manifold
            Instantiated DiscreteSurfaces manifold.
        a0 : float
            First order parameter.
        a1 : float
            Stretching parameter.
        b1 : float
            Shearing parameter.
        c1 : float
            Bending parameter.
        d1 : float
            additonal first order parameter.
        a2 : float
            Second order parameter.
        References
        ----------
        .. [HSKCB2022] "Elastic shape analysis of surfaces with second-order Sobolev metrics: a comprehensive numerical framework", arXiv:2204.04238 [cs.CV], 25 Sep 2022
        """
        self.a0 = a0
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.d1 = d1
        self.a2 = a2
        self.space = space
        self.time_steps = 5

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.
        Parameters
        ----------
        tangent_vec_a: array-like, shape=[nV, 3]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[nV, dim]
            Tangent vector at base point.
        base_point: array-like, shape=[nV, dim]
            Base point.
        Returns
        -------
        inner_product : float
            Inner-product.
        """
        h = tangent_vec_a
        k = tangent_vec_b
        point1 = base_point + h
        point2 = base_point + k
        norm = 0
        if self.a0 > 0 or self.a2 > 0:
            v_areas = self.space.get_vertex_areas(base_point)
            if self.a2 > 0:
                Del = self.space.get_laplacian(base_point)
                norm += self.a2 * gs.sum(
                    gs.einsum("bi,bi->b", Del(h), Del(k)) / v_areas
                )
            if self.a0 > 0:
                norm += self.a0 * gs.sum(v_areas * gs.einsum("bi,bi->b", h, k))
        if self.a1 > 0 or self.b1 > 0 or self.c1 > 0 or self.b1 > 0:
            alpha = self.space.get_mesh_one_forms(base_point)
            g = gs.matmul(gs.transpose(alpha, axes=(0, 2, 1)), alpha)
            areas = gs.sqrt(gs.linalg.det(g))
            n = self.space.get_normal(base_point)
            if self.c1 > 0:
                dn1 = self.space.get_normal(point1) - n
                dn2 = self.space.get_normal(point2) - n
                norm += self.c1 * gs.sum(gs.einsum("bi,bi->b", dn1, dn2) * areas)
            if self.d1 > 0 or self.b1 > 0 or self.a1 > 0:
                ginv = gs.linalg.inv(g)
                alpha1 = self.space.get_mesh_one_forms(point1)
                alpha2 = self.space.get_mesh_one_forms(point2)
                if self.d1 > 0:
                    xi1 = alpha1 - alpha
                    xi1_0 = gs.matmul(
                        gs.matmul(alpha, ginv),
                        gs.matmul(gs.transpose(xi1, (0, 2, 1)), alpha)
                        - gs.matmul(gs.transpose(alpha, axes=(1, 2)), xi1),
                    )
                    xi2 = alpha2 - alpha
                    xi2_0 = gs.matmul(
                        gs.matmul(alpha, ginv),
                        gs.matmul(gs.transpose(xi2, (0, 2, 1)), alpha)
                        - gs.matmul(gs.transpose(alpha, axes=(1, 2)), xi2),
                    )
                    norm += self.d1 * gs.sum(
                        gs.einsum(
                            "bii->b",
                            gs.matmul(
                                xi1_0,
                                gs.matmul(ginv, gs.transpose(xi2_0, axes=(0, 2, 1))),
                            ),
                        )
                        * areas
                    )
                if self.b1 > 0 or self.a1 > 0:
                    dg1 = gs.matmul(gs.transpose(alpha1, axes=(0, 2, 1)), alpha1) - g
                    dg2 = gs.matmul(gs.transpose(alpha2, axes=(0, 2, 1)), alpha2) - g
                    ginvdg1 = gs.matmul(ginv, dg1)
                    ginvdg2 = gs.matmul(ginv, dg2)
                    norm += self.a1 * gs.sum(
                        gs.einsum("bii->b", gs.matmul(ginvdg1, ginvdg2)) * areas
                    )
                    norm += self.b1 * gs.sum(
                        gs.einsum("bii->b", ginvdg1)
                        * gs.einsum("bii->b", ginvdg2)
                        * areas
                    )
        return norm

    def squared_norm(self, vector, base_point):
        """Squared norm of a tangent vector at a base point.
        Parameters
        ----------
        vector: array-like, shape=[nV, 3]
            Tangent vector at base point.
        base_point: array-like, shape=[nV, dim]
            Base point.
        Returns
        -------
        squared_norm : float
            Squared Norm.
        """
        return self.inner_product(vector, vector, base_point)

    def stepwise_path_energy(self, path):
        """Stepwise path energy of a PL path in the space of discrete surfaces.
        Parameters
        ----------
        path: array-like, shape=[time_steps,nV, 3]
            PL path of discrete surfaces.
        Returns
        -------
        stepwise_path_energy : array-like, shape=[time_steps-1]
            Stepwise path energy.
        """
        N = path.shape[0]
        diff = path[1:, :, :] - path[:-1, :, :]
        midpoints = path[0 : N - 1, :, :] + diff / 2
        enr = []
        for i in range(0, N - 1):
            enr += [N * self.squared_norm(diff[i], midpoints[i])]
        return gs.array(enr)

    def path_energy(self, path):
        """Path energy of a PL path in the space of discrete surfaces.
        Parameters
        ----------
        path: array-like, shape=[time_steps,nV, 3]
            PL path of discrete surfaces.
        Returns
        -------
        path_energy : float
            total path energy.
        """
        return 0.5 * gs.sum(self.stepwise_path_energy(path))

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Geodesic: given an initial point and either an endpoint or initial vector computes the geodesic
        Parameters
        ----------
        initial_point: array-like, shape=[nV, 3]
            Initial discrete surface
        end_point: array-like, shape=[nV, 3]
            End discrete surface: endpoint for the boundary value geodesic problem
            Optional, default: None.
        initial_tangent_vec: array-like, shape=[nV, 3]
            Initial tangent vector
            Optional, default: None.
        Returns
        -------
        path_energy : float
            total path energy.
        """
        if end_point is not None:
            return self._bvp(initial_point, end_point)
        if initial_tangent_vec is not None:
            return _ivp(initial_point, initial_tangent_vec)

    def exp(self, tangent_vec, base_point):
        """Exponential map associated to the Riemmannian metric.
        Exponential map at base_point of tangent_vec computed by discrete geodesc calculus methods
        Parameters
        ----------
        tangent_vec : array-like, shape=[nV, 3]
            Tangent vector at the base point.
        base_point : array-like, shape=[nV, 3]
            Point on the manifold.
        Returns
        -------
        exp : array-like, shape=[nv,3]
            Point on the manifold.
        """
        geod = self._ivp(base_point, tangent_vec)
        return geod[-1]

    def log(self, point, base_point):
        """Compute logarithm map associated to the Riemannian metric.
        Solve the boundary value problem associated to the geodesic equation using path straightening.
        Parameters
        ----------
        point : array-like, shape=[nV,3]
            Point on the manifold.
        base_point : array-like, shape=[nV,3]
            Point on the manifold.
        Returns
        -------
        tangent_vec : array-like, shape=[nV,3]
            Tangent vector at the base point.
        """
        geod = self._bvp(base_point, point)
        return (geod[1] - geod[0]) * (time_steps - 1)

    def _bvp(self, initial_point, end_point):
        npoints = initial_point.shape[0]
        step = (end_point - initial_point) / (self.time_steps - 1)
        geod = gs.array([initial_point + i * step for i in range(0, self.time_steps)])
        midpoints = geod[1 : self.time_steps - 1]

        def funopt(midpoint):
            midpoint = gs.reshape(gs.array(midpoint), (self.time_steps - 2, npoints, 3))
            return self.path_energy(
                gs.concatenate(
                    [initial_point[None, :, :], midpoint, end_point[None, :, :]], axis=0
                )
            )

        sol = minimize(
            gs.autodiff.value_and_grad(funopt),
            gs.flatten(midpoints),
            method="L-BFGS-B",
            jac=True,
            options={"disp": True, "ftol": 0.001},
        )
        out = gs.reshape(gs.array(sol.x), (self.time_steps - 2, npoints, 3))
        geod = gs.concatenate(
            [initial_point[None, :, :], out, end_point[None, :, :]], axis=0
        )
        return geod

    def _ivp(self, initial_point, initial_tangent_vec):
        h0 = initial_tangent_vec / (self.time_steps - 1)
        V0 = initial_point
        V1 = V0 + h0
        ivp = [V0, V1]
        for i in range(2, self.time_steps):
            V2 = self._stepforward(V0, V1)
            ivp += [V2]
            V0 = V1
            V1 = V2
        return gs.stack(ivp, axis=0)

    def _stepforward(self, V0, V1):
        npoints = V0.shape[0]
        B = gs.zeros([npoints, 3]).requires_grad_(True)
        qV1 = V1.clone().requires_grad_(True)

        def energy(V2):
            V1dot = V1 - V0
            V2dot = V2 - V1

            def getMetric1(Vdot):
                return self.inner_product(V1dot, Vdot, V0)

            def getMetric2(Vdot):
                return self.inner_product(V2dot, Vdot, V1)

            def norm(V1):
                return self.squared_norm(V2dot, V1)

            sys1 = grad(getMetric1(B), B, create_graph=True)[0]
            sys2 = grad(getMetric2(B), B, create_graph=True)[0]
            sys3 = grad(norm(qV1), qV1, create_graph=True)[0]

            sys = 2 * sys1 - 2 * sys2 + sys3
            return gs.sum(sys**2)

        def funopt(V2):
            V2 = gs.reshape(gs.array(V2), (npoints, 3))
            return energy(V2)

        sol = minimize(
            gs.autodiff.value_and_grad(funopt),
            gs.flatten(2 * (V1 - V0) + V0),
            method="L-BFGS-B",
            jac=True,
            options={"disp": True, "ftol": 0.00001},
        )
        return gs.reshape(gs.array(sol.x), (npoints, 3))

    def dist(self, point_a, point_b):
        """Geodesic distance between two discrete surfaces.

        Parameters
        ----------
        point_a : array-like, shape=[nV,3]
            Point.
        point_b : array-like, shape=[nV,3]
            Point.
        Returns
        -------
        dist : float
            Distance.

        """
        geod = self._bvp(initial_point, end_point)
        enr = self.stepwise_path_energy(geod)
        return gs.sum(gs.sqrt(enr))
