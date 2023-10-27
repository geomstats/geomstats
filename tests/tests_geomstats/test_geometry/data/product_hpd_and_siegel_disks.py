from .product_manifold import ProductManifoldTestData, ProductRiemannianMetricTestData


class ProductHPDMatricesAndSiegelDisksTestData(ProductManifoldTestData):
    pass


class ProductHPDMatricesAndSiegelDisksMetricTestData(ProductRiemannianMetricTestData):
    trials = 5

    tolerances = {
        "dist_is_log_norm": {"atol": 1e-6},
        "geodesic_bvp_reverse": {"atol": 1e-6},
        "geodesic_ivp_belongs": {"atol": 1e-6},
        "exp_belongs": {"atol": 1e-6},
    }
