from .product_manifold import ProductManifoldTestData, ProductRiemannianMetricTestData


class ProductPositiveRealsAndComplexPoincareDisksTestData(ProductManifoldTestData):
    pass


class ProductPositiveRealsAndComplexPoincareDisksMetricTestData(
    ProductRiemannianMetricTestData
):
    trials = 3

    skips = ProductRiemannianMetricTestData.skips + (
        "inner_product_is_symmetric",  # complex part sign problem
    )

    tolerances = {
        "geodesic_ivp_belongs": {"atol": 1e-6},
    }
