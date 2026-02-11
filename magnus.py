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

    def forward(self, t, X):
        """
        X: (3*K,) flattened state
        returns dX/dt
        """
        R = X.view(self.K, 3)

        rp = self.curve.curve_d1(t).squeeze(0)

        dR_list = [rp]

        # If Omega is our solution, then Omega' has a simple form
        # (Omega_n)' = sum over {i_k} which sum to (n-1) [Omage_i_1, [Omega_i_2[...[Omega_i_k, A]...]]]
        # So if we have i_1 = K then the remaining i_k have to sum to (n-1)-K
        # but this is exactly (Omega_(n-1-K))
        # Thus we get (Omega_n)' = sum_(k=1)^(n-1) [Omega_k, (Omega_(n-k))']
        # In the curve domain we replace these commutators with cross product and get our ODE with variables Omega_n
        for k in range(1, self.K):
            acc = torch.zeros(3, device=X.device)

            for j in range(1, k+1):
                coeff = self.magnus_coeff[j]/factorial(j)
                if coeff == 0:
                    continue

                acc = acc + coeff * torch.linalg.cross(
                    R[j-1],
                    dR_list[k-j] # Use previously computed dR from the list
                )

            dR_list.append(acc) # Add the current dR_k to the list

        dR = torch.stack(dR_list) # Stack all components to form the final dR tensor

        return dR.flatten()


def compute_R_terminal(curve, K, T=1.0, steps=2):
    ode = MagnusODE(curve, K)
    X0 = torch.zeros(3*K)

    t = torch.linspace(0.0, T, steps)

    #We only care about evaluating at the final time T which by FToC means
    #we only need to evaluate each Omega_n at time 0 and time T
    XT = odeint_adjoint(ode, X0, t)[-1]

    return XT.view(K, 3)
