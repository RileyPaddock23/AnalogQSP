import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from scipy.special import bernoulli, comb,factorial
import numpy as np

class MagnusODE(nn.Module):
    def __init__(self, curve, K):
        super().__init__()
        self.curve = curve  # must implement curve_d1(t) -> (3,)
        self.K = K
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
        dR[0] = self.curve.curve_d1(t)

        for k in range(2, self.K+1):

            total = 0.0

            for m in range(1, k):

                if self.magnus_coeff[m] == 0:
                    continue

                coeff = self.magnus_coeff[m] / factorial(m)

                for comp in self.compositions(k-1, m):

                    X = self.curve.curve_d1(t)

                    # apply nested ad operators
                    for idx in reversed(comp):
                        X = 2.0*torch.cross(R[idx-1], X)

                    total = total + coeff * X

            dR[k-1] = total

        return dR

def compute_R_terminal(curve, K, T=1.0, steps=2):
    ode = MagnusODE(curve, K)
    X0 = torch.zeros(3*K)

    t = torch.linspace(0.0, T, steps)

    #We only care about evaluating at the final time T which by FToC means
    #we only need to evaluate each Omega_n at time 0 and time T
    XT = odeint_adjoint(ode, X0, t)[-1]

    return XT.view(K, 3)
