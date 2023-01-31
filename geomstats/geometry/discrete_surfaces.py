"""Discrete Surfaces with Elastic metrics.

Lead author: Emmanuel Hartman

This file is on-going work to clean the Geomstats GitHub PR by Emmanuel.
"""

from scipy.optimize import minimize
from torch.autograd import grad

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class DiscreteSurfaces(Manifold):
    r"""Space of parameterized discrete surfaces.

    Each surface is sampled with fixed n_vertices vertices and n_faces faces
    in $\mathbb{R}^3$.

    Each individual surface is represented by a 2d-array of shape `[
    n_vertices, 3]`. This space corresponds to the space of immersions
    defined below, i.e. the
    space of smooth functions from a template to manifold $M$ into  $\mathbb{R}^3$,
    with non-vanishing Jacobian.
    .. math::
        Imm(M,\mathbb{R}^3)=\{ f \in C^{\infty}(M, \mathbb{R}^3)
        \|Df(x)\|\neq 0 \forall x \in M \}.

    Parameters
    ----------
    faces : integer array-like, shape=[n_faces, 3]

    Attributes
    ----------
    faces : integer array-like, shape=[n_faces, 3]
    """

    def __init__(self, faces, **kwargs):
        """Create an object."""
        ambient_dim = 3
        self.dim = (gs.amax(faces) + 1) * ambient_dim
        self.faces = faces
        self.n_faces = len(faces)
        self.n_vertices = int(gs.amax(self.faces) + 1)

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Checks that vertices are inputed in proper form and are
        consistent with the mesh structure.

        Also checks if the discrete surface has degenerate triangles.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
            Surface, i.e. the vertices of its triangulation.
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
        if point.shape[-2] != self.n_vertices:
            return False
        # to make sure that it is actually a face and not just a point
        # or an edge. small area = degenerate face
        smallest_area = min(self.face_areas(point))
        if smallest_area < atol:
            return False
        return True

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., n_vertices, 3]
            Vector.
        base_point : array-like, shape=[..., n_vertices, 3]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return vector.shape[-1] == 3 and vector.shape[-2] == self.n_vertices

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.

        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        raise NotImplementedError("to tangent is not implemented for discrete surfaces")

    def projection(self, point):
        """Project a point to the manifold.

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        raise NotImplementedError("projection is not implemented for discrete surfaces")

    def random_point(self, n_samples=1):
        """Sample discrete surfaces.

        This sample random discrete surfaces with the correct number of vertices.

        Parameters
        ----------
        n_samples : int
            Number of surfaces to sample.
            Optional, Default=1

        Returns
        -------
        vertices :  array-like, shape=[n_samples, n_vertices, 3]
            Vertices for a batch of points in the space of discrete surfaces.
        """
        sphere = Hypersphere(dim=2, default_coords_type="extrinsic")
        vertices = sphere.random_uniform(n_samples * self.n_vertices)
        vertices = gs.reshape(vertices, (n_samples, self.n_vertices, 3))
        return vertices

    def vertex_areas(self, point):
        """Compute vertex areas for a triangulated surface.

        Parameters
        ----------
        point : array-like, shape=[n_verticesx3]
             Surface, i.e. the vertices of its triangulation.

        Returns
        -------
        vertareas :  array-like, shape=[n_verticesx1]
            vertex areas
        """
        n_vertices = point.shape[0]
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
        incident_areas = gs.zeros(n_vertices)
        val = gs.flatten(gs.stack([area] * 3, axis=1))
        incident_areas.scatter_add_(0, idx, val)
        vertAreas = 2 * incident_areas / 3.0
        return vertAreas

    def get_laplacian(self, point):
        """Compute the mesh Laplacian operator of a surface.

        The laplacian is evaluated at one of its tangent vectors h.

        Parameters
        ----------
        point  :  array-like, shape=[n_verticesx3]
             Surface, i.e. the vertices of its triangulation.

        Returns
        -------
        L : callable
            Function that will evaluate the mesh Laplacian operator
            at a tangent vector to the surface
        """
        n_vertices, n_faces = point.shape[0], self.faces.shape[0]
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
        idx = gs.reshape(gs.stack([ii, jj], axis=0), (2, n_faces * 3))

        def L(h):
            """Evaluate the mesh Laplacian operator.

            The operator is evaluated at a tangent vector to the surface.

            Parameters
            ----------
            h :  array-like, shape=[n_verticesx3]
                Tangent vector to the triangulated surface.

            Returns
            -------
            Lh: array-like, shape=[n_verticesx3]
                Mesh Laplacian operator of the triangulated surface applied
                 to one its tangent vector h.
            """
            hdiff = h[idx[0]] - h[idx[1]]
            values = gs.stack([gs.flatten(cot)] * 3, axis=1) * hdiff
            Lh = gs.zeros((n_vertices, 3))
            Lh[:, 0] = Lh[:, 0].scatter_add(0, idx[1, :], values[:, 0])
            Lh[:, 1] = Lh[:, 1].scatter_add(0, idx[1, :], values[:, 1])
            Lh[:, 2] = Lh[:, 2].scatter_add(0, idx[1, :], values[:, 2])
            return Lh

        return L

    def normals(self, point):
        """Compute normals at each face of a triangulated surface.

        Normals are the cross products between edges of each face
        that are incident to its x-coordinate.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
            Surface, i.e. the vertices of its triangulation.

        Returns
        -------
        N : array-like, shape=[n_facesx1]
            Normals of each face of the mesh.
        """
        V0, V1, V2 = (
            gs.take(point, indices=self.faces[:, 0], axis=0),
            gs.take(point, indices=self.faces[:, 1], axis=0),
            gs.take(point, indices=self.faces[:, 2], axis=0),
        )
        N = 0.5 * gs.cross(V1 - V0, V2 - V0)
        return N

    def surface_one_forms(self, point):
        """Compute the vector valued one-forms.

        The one forms are evaluated at the faces of a triangulated surface.

        Parameters
        ----------
        point :  array-like, shape=[n_vertices, 3]
             One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        alpha : array-like, shape=[n_faces, 3, 2]
            One form evaluated at each face of the triangulated surface.
        """
        # TODO: have ambient dimension be last.
        V0, V1, V2 = (
            gs.take(point, indices=self.faces[:, 0], axis=0),
            gs.take(point, indices=self.faces[:, 1], axis=0),
            gs.take(point, indices=self.faces[:, 2], axis=0),
        )
        return gs.stack([V1 - V0, V2 - V0], axis=1)

    def face_areas(self, point):
        """Compute the areas for each face of a triangulated surface.

        The corresponds to the volume area for the surface metric, that is
        the volume area of the pullback metric of the immersion defining the
        surface metric.

        Parameters
        ----------
        point :  array-like, shape=[n_vertices, 3]
            One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        _ :  array-like, shape=[n_faces,]
            Area computed at each face of the triangulated surface.
        """
        g = self.surface_metric_matrices(point)
        return gs.sqrt(gs.linalg.det(g))

    def surface_metric_matrices(self, point):
        """Compute the surface metric matrices.

        The matrices are evaluated at the faces of a triangulated surface.

        The surface metric is the pullback metric of the immersion q
        defining the surface, i.e. of
        the map q: M -> R3, where M is the parameterization manifold.

        Parameters
        ----------
        point : array like, shape=[n_verticesx3]
            One surface, i.e. the vertices of its triangulation.

        Returns
        -------
        _ : array-like, shape=[n_faces, 2, 2]
            Surface metric matrices evaluated at each face of
            the triangulated surface.
        """
        one_forms = self.surface_one_forms(point)
        transposed_one_forms = gs.transpose(one_forms, axes=(0, 2, 1))
        return gs.matmul(one_forms, transposed_one_forms)


class ElasticMetric(RiemannianMetric):
    """Elastic metric defined a family of second order Sobolev metrics.

    Each individual surface is represented by a 2d-array of shape `[
    n_vertices, 3]`.

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
    .. [HSKCB2022] "Elastic shape analysis of surfaces with second-order
    Sobolev metrics: a comprehensive numerical framework",
    arXiv:2204.04238 [cs.CV], 25 Sep 2022
    """

    def __init__(self, space, a0, a1, b1, c1, d1, a2):
        """Create a metric object."""
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
        tangent_vec_a: array-like, shape=[n_vertices, 3]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[n_vertices, dim]
            Tangent vector at base point.
        base_point: array-like, shape=[n_vertices, dim]
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
            v_areas = self.space.vertex_areas(base_point)
            if self.a2 > 0:
                Del = self.space.get_laplacian(base_point)
                norm += self.a2 * gs.sum(
                    gs.einsum("bi,bi->b", Del(h), Del(k)) / v_areas
                )
            if self.a0 > 0:
                norm += self.a0 * gs.sum(v_areas * gs.einsum("bi,bi->b", h, k))
        if self.a1 > 0 or self.b1 > 0 or self.c1 > 0 or self.b1 > 0:
            alpha = self.space.surface_one_forms(base_point)
            g = gs.matmul(gs.transpose(alpha, axes=(0, 2, 1)), alpha)
            areas = gs.sqrt(gs.linalg.det(g))
            n = self.space.normals(base_point)
            if self.c1 > 0:
                dn1 = self.space.normals(point1) - n
                dn2 = self.space.normals(point2) - n
                norm += self.c1 * gs.sum(gs.einsum("bi,bi->b", dn1, dn2) * areas)
            if self.d1 > 0 or self.b1 > 0 or self.a1 > 0:
                ginv = gs.linalg.inv(g)
                alpha1 = self.space.surface_one_forms(point1)
                alpha2 = self.space.surface_one_forms(point2)
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
        vector: array-like, shape=[n_vertices, 3]
            Tangent vector at base point.
        base_point: array-like, shape=[n_vertices, dim]
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
        path: array-like, shape=[time_steps,n_vertices, 3]
            PL path of discrete surfaces.

        Returns
        -------
        stepwise_path_energy : array-like, shape=[time_steps-1]
            Stepwise path energy.
        """
        N = path.shape[0]
        diff = path[1:, :, :] - path[:-1, :, :]
        midpoints = path[0 : N - 1, :, :] + diff / 2  # NOQA
        enr = []
        for i in range(0, N - 1):
            enr += [N * self.squared_norm(diff[i], midpoints[i])]
        return gs.array(enr)

    def path_energy(self, path):
        """Path energy of a PL path in the space of discrete surfaces.

        Parameters
        ----------
        path: array-like, shape=[time_steps,n_vertices, 3]
            PL path of discrete surfaces.

        Returns
        -------
        path_energy : float
            total path energy.
        """
        return 0.5 * gs.sum(self.stepwise_path_energy(path))

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Compute a geodesic.

        Given an initial point and either an endpoint or initial vector.

        Parameters
        ----------
        initial_point: array-like, shape=[n_vertices, 3]
            Initial discrete surface
        end_point: array-like, shape=[n_vertices, 3]
            End discrete surface: endpoint for the boundary value geodesic problem
            Optional, default: None.
        initial_tangent_vec: array-like, shape=[n_vertices, 3]
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
            return self._ivp(initial_point, initial_tangent_vec)

    def exp(self, tangent_vec, base_point):
        """Exponential map associated to the Riemmannian metric.

        Exponential map at base_point of tangent_vec computed
        by discrete geodesic calculus methods.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_vertices, 3]
            Tangent vector at the base point.
        base_point : array-like, shape=[n_vertices, 3]
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

        Solve the boundary value problem associated to the geodesic equation
        using path straightening.

        Parameters
        ----------
        point : array-like, shape=[n_vertices,3]
            Point on the manifold.
        base_point : array-like, shape=[n_vertices,3]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[n_vertices,3]
            Tangent vector at the base point.
        """
        geod = self._bvp(base_point, point)
        return geod[1] - geod[0]

    def _bvp(self, initial_point, end_point):
        npoints = initial_point.shape[0]
        step = (end_point - initial_point) / (self.time_steps - 1)
        geod = gs.array([initial_point + i * step for i in range(0, self.time_steps)])
        midpoints = geod[1 : self.time_steps - 1]  # NOQA

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
        point_a : array-like, shape=[n_vertices,3]
            Point.
        point_b : array-like, shape=[n_vertices,3]
            Point.

        Returns
        -------
        dist : float
            Distance.
        """
        geod = self._bvp(point_a, point_b)
        enr = self.stepwise_path_energy(geod)
        return gs.sum(gs.sqrt(enr))
