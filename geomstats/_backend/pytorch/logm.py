import torch
from ..numpy import linalg as nplinalg

def _adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)

def _logm_geomstats(A):
    return torch.from_numpy(nplinalg.logm(A.cpu())).to(A.device)

class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return _logm_geomstats(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return _adjoint(A, G, _logm_geomstats)

logm = Logm.apply
