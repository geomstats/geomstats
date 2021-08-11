"""Classes for the pullback metric."""

import autograd

from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PullbackMetric(RiemannianMetric):
    r"""Pullback metric.

    Let :math:`f` be an immersion :math:`f: M \rightarrow N`
    of one manifold :math:`M` into the Riemannian manifold :math:`N`
    with metric :math:`g`.
    The pull-back metric :math:`f^*g` is defined on :math:`M` for a
    base point :math:`p` as:
    :math:`(f^*g)_p(u, v) = g_{f(p)}(df_p u , df_p v)
    \quad \forall u, v \in T_pM`

    Note
    ----
    The pull-back metric is currently only implemented for an
    immersion into the Euclidean space, i.e. for 
    :math:`N=\mathbb{R}^n`.

    Parameters
    ----------
    dim : int
        Dimension of the underlying manifold.
    embedding_dim : int
        Dimension of the embedding Euclidean space.
    immersion : callable
        Map defining the immersion into the Euclidean space.
    """
    def __init__(self, dim, embedding_dim, immersion, tangent_immersion=None):
        super(PullbackMetric, self).__init__(dim=dim)
        self.embedding_metric = EuclideanMetric(embedding_dim)       
        self.immersion = immersion
        if tangent_immersion is None:
            tangent_immersion = autograd.jacobian(immersion)
        self.tangent_immersion = tangent_immersion

    def metric_matrix(self, base_point=None):
        r"""Metric matrix at the tangent space at a base point.

        Let :math:`f` be the immersion 
        :math:`f: M \rightarrow \mathbb{R}^n` of the manifold
       :math:`M` into the Euclidean space :math:`\mathbb{R}^n`.

       The elements of the metric matrix at a base point :math:`p`
       are defined as:
       :math:`(f*g)_{ij}(p) = <df_p e_i , df_p e_j>`,
       for :math:`e_i, e_j` basis elements of :math:`M`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        metric_mat = gs.zeros((self.dim,) * 2)
        immersed_base_point = self.immersion(base_point)
        all_lines = []
        for i in range(self.dim):
            line_i = []
            basis_element_i = gs.zeros_like(base_point)
            basis_element_i[i] = 1.
            immersed_basis_element_i = gs.matmul(
                self.tangent_immersion(base_point), basis_element_i)
            for j in range(self.dim):
                basis_element_j = gs.zeros_like(base_point)
                basis_element_j[j] = 1.
                immersed_basis_element_j = gs.matmul(
                    self.tangent_immersion(base_point), basis_element_j)
                
                value = self.embedding_metric.inner_product(
                    immersed_basis_element_i, 
                    immersed_basis_element_j, 
                    base_point=immersed_base_point)
                line_i.append(value)
            all_lines.append(gs.array(line_i))

        return gs.vstack(all_lines)