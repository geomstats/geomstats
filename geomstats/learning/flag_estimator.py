from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2

import geomstats.backend as gs
from geomstats.geometry.grassmannian import GrassmannianCanonicalMetric
from geomstats.geometry.matrices import Matrices


class FlagEstimator:
    def __init__(self, multiplicities):
        self.multiplicities = multiplicities
        self.d = sum(self.multiplicities)
        self.n_eig = len(self.multiplicities)

    def _get_P_j(self, j):
        mult_j = self.multiplicities[j]

        k = sum(self.multiplicities[:j])
        indices = [(k + i, k + i) for i in range(mult_j)]
        return gs.array_from_sparse(
            data=[1 for _ in range(mult_j)],
            indices=indices,
            target_shape=(self.d, self.d),
        )

    def _get_projector_j(self, j, vec_emp):
        mult_j = self.multiplicities[j]
        k = sum(self.multiplicities[:j])

        U_j = vec_emp[:, k: k + mult_j]

        return gs.sum(gs.einsum("i..., j...->...ij", U_j, U_j), axis=0)

    def _get_sigma_j(self, j, val_emp_means):
        sigma_j = []
        for k in range(self.n_eig):
            if j == k:
                val = 1.0
            else:
                lambda_j = val_emp_means[j]
                lambda_k = val_emp_means[k]
                val = gs.sqrt(lambda_j * lambda_k / (lambda_j - lambda_k) ** 2)
            sigma_j.append(val)

        return sigma_j

    def _get_K_j(self, j, sigma_j):
        diag_indices = [(i, i) for i in range(self.d)]

        indices = []
        for i, mult in enumerate(self.multiplicities):
            indices.extend([i] * mult)

        sigma_j_ = gs.take(sigma_j, indices)

        values = 1.0 / sigma_j_
        return gs.array_from_sparse(diag_indices, values, (self.d, self.d))

    def _is_in_cut_locus_j(self, j, vec_emp, tol=gs.atol):

        mult_j = self.multiplicities[j]
        k = sum(self.multiplicities[:j])
        E_jj = vec_emp[k: k + mult_j, k: k + mult_j]
        det_E_jj = gs.linalg.det(E_jj)

        return gs.abs(det_E_jj) < tol

    def _get_empirical_cov(self, X):
        return EmpiricalCovariance().fit(X).covariance_

    def _compute_eigh(self, mat, reverse=True):
        val, vec = gs.linalg.eigh(mat)

        if reverse:
            val = gs.flip(val)
            vec = gs.flip(vec, axis=1)

        return val, vec

    def _get_val_emp_means(self, val_emp):
        val_emp_means = []
        for j, mult_j in enumerate(self.multiplicities):
            k = sum(self.multiplicities[:j])
            val_emp_means.append(gs.mean(val_emp[k: k + mult_j]))

        return gs.stack(val_emp_means)

    def compute_norm(self, X, Gamma, reverse_eig=True):
        n = X.shape[0]
        cov = self._get_empirical_cov(X)

        val_emp, vec_emp = self._compute_eigh(cov, reverse=reverse_eig)
        val_emp_means = self._get_val_emp_means(val_emp)

        norm = 0
        for j, mult_j in enumerate(self.multiplicities):
            if self._is_in_cut_locus_j(j, vec_emp):
                continue

            metric_j = GrassmannianCanonicalMetric(self.d, mult_j)

            P_j = self._get_projector_j(j, vec_emp)
            P_0_j = self._get_P_j(j)

            log_input = Matrices.mul(gs.transpose(Gamma), P_j, Gamma)

            sigma_j = self._get_sigma_j(j, val_emp_means)
            K_j = self._get_K_j(j, sigma_j)

            out = Matrices.mul(K_j, metric_j.log(log_input, P_0_j), K_j)

            norm += Matrices.frobenius_product(out, out)

        return norm * n / 4.0

    def get_flag(self, Q_0):
        F = []
        for j, _ in enumerate(self.multiplicities):
            P_0_j = self._get_P_j(j)
            F.append(Matrices.mul(Q_0, P_0_j, gs.transpose(Q_0)))

        return gs.stack(F)


def test_H0(estimator, X, Q_0, alpha=.05):
    """Test if Q_0 is a matrix of eigenvectors of the population covariance matrix.

    If True, hypothesis is accepted.
    """
    norm = estimator.compute_norm(X, Q_0)

    df = (estimator.d**2 - gs.sum(gs.array(estimator.multiplicities)**2)) / 2

    quantile = chi2.ppf(1 - alpha, df)

    cmp = norm < quantile

    return cmp, (norm, quantile)
