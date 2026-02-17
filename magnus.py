import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from scipy.special import bernoulli, factorial

class MagnusODE(nn.Module):
    def __init__(self, curve, K):
        super().__init__()
        self.curve = curve  # must implement curve_d1(t) -> (3,)
        self.K = K
        # B_j for j=0..K-1
        self.magnus_coeff = bernoulli(K)
    
    def compositions(self, n, m):
        if m == 1:
            yield (n,)
            return
        for i in range(1, n - m + 2):
            for tail in self.compositions(n - i, m - 1):
                yield (i,) + tail

    def forward(self, t, X):
        """
        X: (3*K,) flattened state
        returns dX/dt
        """
        # def magnus_rhs_terms(R_terms, r_prime, max_order):
        """
        R_terms: list [R1, R2, ..., R_{k-1}]
        r_prime: current r'(t)
        """
        dR = [None] * self.K
        R = X.view(self.K, 3)

        # Order 1
        tangent = self.curve.curve_d1(t)
        if tangent.ndim > 1:
            tangent = tangent.squeeze(0)
        dR[0] = tangent.to(dtype=X.dtype, device=X.device)

        for k in range(2, self.K+1):

            total = torch.zeros(3, dtype=X.dtype, device=X.device)

            for m in range(1, k):

                if self.magnus_coeff[m] == 0:
                    continue

                coeff = self.magnus_coeff[m] / factorial(m)
                prefactor = coeff * ((2j) ** m)

                for comp in self.compositions(k-1, m):

                    term = tangent.to(dtype=X.dtype, device=X.device)

                    # apply nested ad operators
                    for idx in reversed(comp):
                        term = torch.cross(R[idx - 1], term, dim=-1)

                    total = total + prefactor * term

            dR[k-1] = total

        return torch.stack(dR, dim=0).reshape(-1)

def compute_R_terminal(curve, K, T=1.0, steps=2):
    ode = MagnusODE(curve, K)
    tangent0 = curve.curve_d1(torch.as_tensor(0.0))
    if tangent0.ndim > 1:
        tangent0 = tangent0.squeeze(0)

    dtype = torch.complex64 if not torch.is_complex(tangent0) else tangent0.dtype
    device = tangent0.device

    X0 = torch.zeros(3 * K, dtype=dtype, device=device)

    t = torch.linspace(0.0, T, steps, dtype=tangent0.real.dtype, device=device)

    #We only care about evaluating at the final time T which by FToC means
    #we only need to evaluate each Omega_n at time 0 and time T
    XT = odeint_adjoint(ode, X0, t)[-1]

    return XT.view(K, 3)
