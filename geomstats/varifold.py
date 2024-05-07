import geomstats.backend as gs

if gs.__name__.endswith("numpy"):
    from pykeops.numpy import Genred
else:
    from pykeops.torch import Genred


# TODO: rethink all of this based on
# https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html#data-attachment-term

# TODO: delete?
SPATIAL_KERNEL = {
    "gaussian": "Exp(-SqDist(x,y)*a)",
    "cauchy": "IntCst(1)/(IntCst(1)+SqDist(x,y)*a)",
    "energy": "-Clamp(Norm2(x-y),cmin,cmax)",
}

GRASSMANIAN_KERNEL = {
    "constant": "IntCst(1)",
    "linear": "(u|v)",
    "gaussian_oriented": "Exp(IntCst(2)*b*((u|v)-IntCst(1)))",
    "binet": "Square((u|v))",
    "gaussian_unoriented": "Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))",
}

SIGNAL_KERNEL = {
    "constant": "IntCst(1)",
    "gaussian": " Exp(-SqDist(g,h)*c)",
    "cauchy": "IntCst(1)/(IntCst(1)+SqDist(g,h)*c)",
}


class Reduction:
    def __init__(self, formula, aliases=None, fixed_params=None, **reduction_kwargs):
        if aliases is None:
            aliases = {}
        if fixed_params is None:
            fixed_params = {}

        self.formula = formula
        self.aliases = aliases

        self.fixed_params = fixed_params
        self._fixed_params_ls = list(fixed_params.values())

        self.reduction_kwargs = reduction_kwargs
        self._routine = None

    def _generate_routine(self):
        aliases = [f"{key} = Pm(1)" for key in self.fixed_params.keys()]
        aliases += [f"{key} = {value}" for key, value in self.aliases.items()]

        return Genred(self.formula, aliases, **self.reduction_kwargs)

    def __call__(self, *args):
        if self._routine is None:
            self._routine = self._generate_routine()

        return self._routine(*self._fixed_params_ls, *args)


class ConstantReduction(Reduction):
    def __init__(self, value):
        self.value = value
        super().__init__(formula=f"IntCst({value})")


class GaussianKernel(Reduction):
    def __init__(self, dim=3, sigma_param=0.2, x="x", a="a"):
        self.dim = dim
        self.sigma = sigma_param
        formula = f"Exp(-SqDist({x}0,{x}1)*a)"

        super().__init__(
            formula,
            aliases={
                f"{x}{index}": f"V{index_name}({dim})"
                for index, index_name in enumerate("ij")
            },
            fixed_params={"a": gs.array([1 / sigma_param**2])},
            reduction_op="Sum",
            axis=1,
        )


class GaussianOrientedKernel(Reduction):
    def __init__(self, dim=3, b_param=1.0, u="u", b="b"):
        formula = f"Exp(IntCst(2)*{b}*(({u}0|{u}1)-IntCst(1)))"

        super().__init__(
            formula,
            aliases={
                f"{u}{index}": f"V{index_name}({dim})"
                for index, index_name in enumerate("ij")
            },
            fixed_params={"b": gs.array([b_param])},
            reduction_op="Sum",
            axis=1,
        )


class FactorizedReduction(Reduction):
    """A reduction resulting from the multiplications of reductions.

    NB: assumes `reduction_op` and `axis` are the same across reductions.
    """

    def __init__(self, factors):
        self.factors = factors
        formula = "*".join([factor.formula for factor in factors])

        aliases = factors[0].aliases.copy()
        fixed_params = factors[0].fixed_params.copy()
        for factor in factors[1:]:
            aliases.update(factor.aliases)
            fixed_params.update(factor.fixed_params)

        super().__init__(formula, aliases, fixed_params, **factors[0].reduction_kwargs)


class AreaReduction(Reduction):
    def __init__(self, x="A"):
        var_names = [f"{x}{index}" for index in range(2)]
        formula = "*".join(var_names)
        aliases = {
            var_name: f"V{index_name}(1)"
            for var_name, index_name in zip(var_names, "ij")
        }
        super().__init__(formula, aliases)


class VarifoldReduction(FactorizedReduction):
    def __init__(
        self,
        dim=3,
        spatial_kernel=None,
        tangent_kernel=None,
    ):
        if spatial_kernel is None:
            spatial_kernel = GaussianKernel(dim=dim)

        if tangent_kernel is None:
            tangent_kernel = GaussianOrientedKernel(dim=dim)

        area_reduction = AreaReduction()

        factors = [spatial_kernel, tangent_kernel, area_reduction]
        super().__init__(factors)


class VarifoldMetric:
    """Varifold metric."""

    def __init__(self, space, varifold_reduction=None, other_space=None):
        if other_space is None:
            other_space = space

        if varifold_reduction is None:
            self.varifold_reduction = VarifoldReduction()

        self._space = space
        self._other_space = other_space

        self.varifold_reduction = varifold_reduction

    def _compute_mesh_info(
        self, point, centroids=None, normals=None, areas=None, use_other=False
    ):
        space = self._other_space if use_other else self._space

        if centroids is None:
            centroids = space.face_centroids(point)

        if normals is None:
            # TODO: need to normalize normals
            normals = space.normals(point)

        if areas is None:
            areas = space.face_areas(point)

        return centroids, normals, areas

    def scalar_product_per_face(
        self,
        point_a,
        point_b,
        centroids_a=None,
        centroids_b=None,
        normals_a=None,
        normals_b=None,
        areas_a=None,
        areas_b=None,
    ):
        centroids_a, normals_a, areas_a = self._compute_mesh_info(
            point_a, centroids_a, normals_a, areas_a
        )
        centroids_b, normals_b, areas_b = self._compute_mesh_info(
            point_b, centroids_b, normals_b, areas_b, use_other=True
        )

        return self.varifold_reduction(
            centroids_a, centroids_b, normals_a, normals_b, areas_a, areas_b
        )

    def scalar_product(
        self,
        point_a,
        point_b,
        centroids_a=None,
        centroids_b=None,
        normals_a=None,
        normals_b=None,
        areas_a=None,
        areas_b=None,
    ):
        centroids_a, normals_a, areas_a = self._compute_mesh_info(
            point_a, centroids_a, normals_a, areas_a
        )
        centroids_b, normals_b, areas_b = self._compute_mesh_info(
            point_b, centroids_b, normals_b, areas_b, use_other=True
        )

        return gs.sum(
            self.scalar_product_per_face(
                point_a,
                point_b,
                centroids_a,
                centroids_b,
                normals_a,
                normals_b,
                areas_a,
                areas_b,
            ),
            axis=[-2, -1],
        )

    def squared_dist(
        self,
        point_a,
        point_b,
        centroids_a=None,
        centroids_b=None,
        normals_a=None,
        normals_b=None,
        areas_a=None,
        areas_b=None,
    ):
        centroids_a, normals_a, areas_a = self._compute_mesh_info(
            point_a, centroids_a, normals_a, areas_a
        )
        centroids_b, normals_b, areas_b = self._compute_mesh_info(
            point_b, centroids_b, normals_b, areas_b, use_other=True
        )

        # TODO: complete
