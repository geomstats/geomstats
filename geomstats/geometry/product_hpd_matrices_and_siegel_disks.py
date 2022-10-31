"""The ProductHPDMatricesAndSiegelDisks manifold.

The HPD Siegel disks product is defined as a product manifold of the HPD 
manifold and (n-1) Siegel disks. The HPD Siegel disks product has a product 
metric. The product metric on the HPD Siegel disks product space is the usual 
HPD metric and Siegel metrics multiplied by constants.

Lead author: Yann Cabanes.

References
----------
    .. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
      matrices with Toeplitz structured blocks, 2016.
      https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

from geomstats.geometry.hpd_matrices import HPDAffineMetric, HPDMatrices
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric  # NOQA
from geomstats.geometry.siegel import Siegel, SiegelMetric


class ProductHPDMatricesAndSiegelDisks(ProductManifold):
    """Class for the HPD and Siegel product manifold.

    The HPD and Siegel product manifold is a direct product of the HPD manifold
    and (n-1) Siegel disks. Each manifold of the product is a square matrix
    manifold of the same dimension.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.
    n : int
        Size of the matrices.
    """

    def __init__(self, n_manifolds, n, **kwargs):
        # self.default_point_type = default_point_type
        # if default_point_type == "matrix":
        #     self.point_shape = (n_manifolds, n, n)
        #     self.n_dim_point = 3
        # elif default_point_type == "vector":
        #     self.point_shape = (n**2,) * n_manifolds
        #     self.n_dim_point = 1
        self.dim = n_manifolds * n
        self.metric = ProductHPDMatricesAndSiegelDisksMetric(
            n_manifolds=n_manifolds, n=n, **kwargs
        )
        hpd_matrices = HPDMatrices(n=n)
        hpd_matrices.metric = HPDAffineMetric(n=n, scale=n_manifolds**0.5)
        siegel_disk = Siegel(n=n)
        list_manifolds = [hpd_matrices,] + (n_manifolds - 1) * [
            siegel_disk,
        ]
        super(ProductHPDMatricesAndSiegelDisks, self).__init__(
            factors=list_manifolds, **kwargs
        )


class ProductHPDMatricesAndSiegelDisksMetric(ProductRiemannianMetric):
    """Class defining the HPD Siegel disks product metric.

    The HPD Siegel disks product metric is a product of the HPD metric
    and (n-1) Siegel metrics, each of them being multiplied by a specific
    constant factor (see [JV2016]_).
    This metric comes from a model used to represent
    stationary multidimensional complex autoregressive Gaussian signals.

    Parameters
    ----------
    n_manifolds : int
        Number of manifolds of the product.
    n : int
        Size of the matrices.

    References
    ----------
    .. [JV2016] B. Jeuris and R. Vandebril. The Kähler mean of Block-Toeplitz
      matrices with Toeplitz structured blocks, 2016.
      https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n_manifolds, n, **kwargs):
        self.n_manifolds = n_manifolds
        self.n = n
        # self.default_point_type = default_point_type
        # if default_point_type == "matrix":
        #     self.point_shape = (n_manifolds, n, n)
        #     self.n_dim_point = 3
        # elif default_point_type == "vector":
        #     self.point_shape = (n**2,) * n_manifolds
        #     self.n_dim_point = 1
        list_metrics = [
            HPDAffineMetric(n=n, scale=n_manifolds**0.5),
        ]
        for i_space in range(1, n_manifolds):
            scale_i = (n_manifolds - i_space) ** 0.5
            metric_i = SiegelMetric(n=n, scale=scale_i)
            list_metrics.append(metric_i)
        super(ProductHPDMatricesAndSiegelDisksMetric, self).__init__(
            metrics=list_metrics
        )


product_hpd_matrices_and_siegel_disks = ProductHPDMatricesAndSiegelDisks(
    n_manifolds=3, n=2
)
# print(product_hpd_matrices_and_siegel_disks)
#
# point_a = gs.array([
#     [[0.80378, -0.27852 + 0.15j], [-0.27852 - 0.15j, 0.8025522]],
#     [[0.15902535, -0.27802], [0.0427, 0.198702527]],
#     [[0.25945, 0.17857], [-0.45602527, -0.598707272]]])
#
# point_b = gs.array([
#     [[0.82042, 0.352 - 0.212j], [0.352 + 0.212j, 0.602764]],
#     [[-0.3595723, -0.278], [0.2565273, -0.3987]],
#     [[0.359857, -0.27875334], [0.2563344, -0.3987855 - 0.15273j]]])
#
# point_c = gs.array([
#     [[0.720256, -0.2785 + 0.166j], [-0.2785 - 0.166j, 0.80073]],
#     [[0.1577552, -0.274528], [0.456732, 0.3987575]],
#     [[-0.1564, 0.2785635], [-0.256437, -0.598537]]])
#
# point_d = gs.array([
#     [[0.7211, 0.3886 - 0.253j], [0.3886 + 0.253j, 0.6022]],
#     [[-0.15863, -0.178533], [0.35686, -0.1987]],
#     [[0.159, -0.27838], [0.156863, -0.0987 - 0.1864j]]])
#
# point_e = gs.array([
#     [[0.623, -0.278 + 0.1288j], [-0.278 - 0.1288j, 0.75238]],
#     [[-0.3843388, -0.278456], [0.45536, -0.39878634 + 0.37j]],
#     [[0.1438, -0.37853], [0.356686, -0.5987448]]])
#
# point_f = gs.array([
#     [[0.62, 0.3 - 0.2j], [0.3 + 0.2j, 0.6]],
#     [[0.1 + 0.13j, -0.278 - 0.23j], [0.456, -0.3987]],
#     [[0.15528, -0.178], [0.556789, -0.3987 - 0.135j]]])
#
# points = gs.stack(
#     [point_a,
#      point_b,
#      point_c,
#      point_d,
#      point_e,
#      point_f],
#     axis=0)
#
# print(product_hpd_matrices_and_siegel_disks.belongs(points))
#
# print(product_hpd_matrices_and_siegel_disks.metric.squared_dist(point_a=point_a, point_b=points))
#
# logarithms = product_hpd_matrices_and_siegel_disks.metric.log(base_point=point_a, point=points)
#
# print(logarithms.shape)
#
# exp = product_hpd_matrices_and_siegel_disks.metric.exp(base_point=point_a, tangent_vec=logarithms)
#
# print(exp.shape)
#
# sq_norm = product_hpd_matrices_and_siegel_disks.metric.squared_norm(
#     vector=logarithms,
#     base_point=point_a)
#
# print(sq_norm)

#
#
# print(point_a.shape)
# print(points.shape)
#
# print(product_hpd_matrices_and_siegel_disks.belongs(point_a))
# print(product_hpd_matrices_and_siegel_disks.belongs(point_b))


# print(product_hpd_matrices_and_siegel_disks.metric.dist(point_a=point_a, point_b=point_a))
# print('\n')
# print(product_hpd_matrices_and_siegel_disks.metric.dist(point_a=point_a, point_b=points))


# from geometriclearning.geometry.hpd_matrices import HPDInformationGeometryMetric
#
# hpd_information_geometry_metric = HPDInformationGeometryMetric(n=2)
# print(hpd_information_geometry_metric.squared_dist(point_a[0], point_b[0]))
# print(hpd_information_geometry_metric.squared_dist(point_a[0], point_a[0]))
# print(hpd_information_geometry_metric.squared_dist(point_b[0], point_b[0]))

# from geometriclearning.geometry.siegel import Siegel
#
# siegeldisk = Siegel(n=2)
# print(siegeldisk.belongs(point_a[2, ...]))
# print(siegeldisk.belongs(point_b[2, ...]))
# print('\n')
# # print(siegeldisk.metric.squared_dist(point_a[1, ...], point_a[1, ...]))
# # print(siegeldisk.metric.squared_dist(point_a[1, ...], point_b[1, ...]))
# print(siegeldisk.metric.squared_dist(point_a[2, ...], point_b[2, ...]))

# print(product_hpd_matrices_and_siegel_disks.metric.squared_dist(point_a=point_a, point_b=points))
# print(product_hpd_matrices_and_siegel_disks.metric.dist(point_a=point_a, point_b=points))
#
# logarithms = product_hpd_matrices_and_siegel_disks.metric.log(base_point=point_a, point=points)
# print(logarithms)

# exp = product_hpd_matrices_and_siegel_disks.metric.exp(base_point=point_a, tangent_vec=logarithms)
#
# sq_norm = product_hpd_matrices_and_siegel_disks.metric.squared_norm(
#     vector=logarithms,
#     base_point=point_a)
#
# print(sq_norm)
#
# norm = product_hpd_matrices_and_siegel_disks.metric.norm(
#     vector=logarithms,
#     base_point=point_a)
#
# print(norm)
#

# belongs_point_a = product_hpd_matrices_and_siegel_disks.belongs(point_a)
# print(belongs_point_a)
#
# belongs_point_b = product_hpd_matrices_and_siegel_disks.belongs(point_b)
# print(belongs_point_b)
#
# belongs_points = product_hpd_matrices_and_siegel_disks.belongs(points)
# print(belongs_points)

# print(product_hpd_matrices_and_siegel_disks.metric.squared_norm(vector=logarithms, base_point=point_a))
#
# print(product_hpd_matrices_and_siegel_disks.metric.norm(vector=logarithms, base_point=point_a))
# print(product_hpd_matrices_and_siegel_disks.metric.dist(point_a=point_a, point_b=point_b))
# print(product_hpd_matrices_and_siegel_disks.metric.squared_norm(vector=logarithms, base_point=point_a) ** 0.5)

# logarithms = product_hpd_matrices_and_siegel_disks.metric.log(base_point=point_a, point=points)
#
# exp = product_hpd_matrices_and_siegel_disks.metric.exp(base_point=point_a, tangent_vec=logarithms)
#
# # print(exp - point_b)
#
# print(product_hpd_matrices_and_siegel_disks.metric.squared_norm(vector=logarithms, base_point=point_a))

# from geometriclearning.learning.frechet_mean import FrechetMean
#
# frechet_mean_product_hpd_matrices_and_siegel_disks = FrechetMean(
#     metric=product_hpd_matrices_and_siegel_disks.metric,
#     max_iter=1,
#     epsilon=0,
#     verbose=True)
# frechet_mean = frechet_mean_product_hpd_matrices_and_siegel_disks.fit(points)
# product_hpd_matrices_and_siegel_disks_mean = frechet_mean.estimate_
# print(product_hpd_matrices_and_siegel_disks_mean)
# # sq_dist_init = product_hpd_matrices_and_siegel_disks.metric.squared_dist(point_a, point_b)
# # print(sq_dist_init / 4)
#
# print('\n')
#
# frechet_mean_product_hpd_matrices_and_siegel_disks = FrechetMean(
#     metric=product_hpd_matrices_and_siegel_disks.metric,
#     max_iter=1000,
#     epsilon=0,
#     method='stochastic',
#     verbose=True)
# frechet_mean = frechet_mean_product_hpd_matrices_and_siegel_disks.fit(points)
# product_hpd_matrices_and_siegel_disks_mean = frechet_mean.estimate_
# print(product_hpd_matrices_and_siegel_disks_mean)

# print('\n')
# print('\n')
#
# from geometriclearning.learning.kmeans import RiemannianKMeans
#
# kmeans = RiemannianKMeans(
#     metric=product_hpd_matrices_and_siegel_disks.metric,
#     n_clusters=2,
#     mean_method='stochastic',
#     mean_max_iter=50,
#     point_type='matrix',
#     verbose=True)
#
# centroids = kmeans.fit(points, max_iter=10)
#
# print(kmeans)
#
# labels = kmeans.predict(points)
#
# print(labels)

# from geometriclearning.geometry.hpd_matrices import HPDMatrices
#
# hpdmatrices = HPDMatrices(n=2)
# hpdinformationgeometrymetric = HPDInformationGeometryMetric(n=2)
#
# kmeans = RiemannianKMeans(
#     metric=hpdinformationgeometrymetric,
#     n_clusters=2,
#     mean_method='stochastic',
#     mean_max_iter=50,
#     point_type='matrix',
#     verbose=True)
#
# points = points[:, 0, ...]
#
# hpdmatrices = HPDMatrices(n=2)
# print(hpdmatrices.belongs(points))
#
# centroids = kmeans.fit(points, max_iter=10)
#
# print(kmeans)
#
# labels = kmeans.predict(points)
#
# print(labels)


# from geometriclearning.geometry.siegel import Siegel
#
# siegel = Siegel(n=2)
#
# kmeans = RiemannianKMeans(
#     metric=siegel.metric,
#     n_clusters=2,
#     mean_method='stochastic',
#     mean_max_iter=100,
#     point_type='matrix',
#     verbose=True,
#     tol=0.001)
#
# points = points[:, 1, ...]
#
# print(siegel.belongs(points))
#
# centroids = kmeans.fit(points, max_iter=10)
#
# print(kmeans)
#
# labels = kmeans.predict(points)
#
# print(labels)
