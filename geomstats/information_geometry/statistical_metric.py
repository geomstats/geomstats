import geomstats.backend as gs

from geomstats.geometry.riemannian_metric import RiemannianMetric

# from geomstats.geometry.euclidean import Euclidean

# euc_2 = Euclidean(dim=2)
# print(euc_2.metric.christoffels(gs.array([1., 1.])))
# print(euc_2.metric.christoffels(gs.array([1., 1.])).shape)

def bregman_divergence(func):
    def _bregman_divergence(point, base_point):
        grad_func = gs.autodiff.value_and_grad(func)
        func_basepoint, grad_func_basepoint = grad_func(base_point)
        bregman_div = func(point) - func_basepoint - gs.dot(point - base_point, grad_func_basepoint)
        return bregman_div
    return _bregman_divergence 

class StatisticalMetric(RiemannianMetric):

    def __init__(self, dim, divergence):
        self.dim = dim
        self.divergence = self._unpack_tensor(divergence)
        print(self.divergence(gs.array([0.0, 0.0, 0.0, 0.0])))

    def _unpack_tensor(self, func):
        def wrapper(tensor):
            return func(tensor[...,:self.dim], tensor[...,self.dim:])
        return wrapper

    def metric_matrix(self,base_point):
        hess = gs.autodiff.hessian(self.divergence)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1*hess(base_point_pair)[:self.dim, self.dim:]
        
    def divergence_christoffels(self, base_point):
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1*jac_hess(base_point_pair)[:2, :2, 2:]
    
    def dual_divergence_christoffels(self, base_point):
        hess = gs.autodiff.hessian(self.divergence)
        jac_hess = gs.autodiff.jacobian(hess)
        base_point_pair = gs.concatenate([base_point, base_point])
        return -1*jac_hess(base_point_pair)[:2, 2:, 2:]
    
    def amari_divergence_tensor(self, base_point):
        divergence_christoffels = self.divergence_christoffels(base_point)
        dual_divergence_christoffels = self.dual_divergence_christoffels(base_point)
        return dual_divergence_christoffels - divergence_christoffels

breg_div = bregman_divergence(lambda x: gs.sum(x**4))

statmetric = StatisticalMetric(dim=2, divergence=breg_div)

# print(statmetric.amari_divergence_tensor(base_point=gs.array([1.0, 1.0])))
# jac_1 = gs.autodiff.jacobian(lambda x: gs.sum(x**4))
# jac_2 = gs.autodiff.jacobian(jac_1)
# jac_3 = gs.autodiff.jacobian(jac_2)
# print(jac_3(gs.array([1.,1.])))

print(statmetric.metric_matrix(gs.array([1.,1.])))
# print(gs.autodiff.hessian(lambda x: gs.sum(x**2))(gs.array([1., 1.]) ) )

# hess = gs.autodiff.hessian(breg_div)
# # print(hess(gs.array([1., 1., 1., 1.])))

# third_der = gs.autodiff.jacobian(hess)

# print(third_der(gs.array([1., 1., 1., 1.]))[:2, :2, 2:])

# print(gs.autodiff.hessian(lambda x: gs.sum(x**2))(gs.array([1., 1.]) ) )