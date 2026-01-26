from base_curve import ArclenParameterize, LearnableCurve
import torch
import torch.nn as nn
from typing import Callable

class BezierCurve(ArclenParameterize):
    def __init__(self, n_ctrl: int, P_T: torch.Tensor):
        assert n_ctrl >= 6
        super().__init__(interval=[0, 1])

        self.n_ctrl = n_ctrl
        self.dim = P_T.shape[-1]

        self.register_buffer('P_T', P_T)

        self.P1 = nn.Parameter(torch.randn(self.dim))
        self.Pn2 = nn.Parameter(torch.randn(self.dim))

        n_interior = n_ctrl - 6
        if n_interior > 0: self.interior = nn.Parameter(torch.randn(n_interior, self.dim))
        else: self.interior = None

    def control_points(self):
        P0 = torch.zeros_like(self.P_T)
        P1 = self.P1
        P2 = 2.0 * P1

        Pn2 = self.Pn2
        Pn3 = 2.0 * Pn2 - self.P_T
        Pn1 = self.P_T

        if self.interior is None: pts = [ P0, P1, P2, Pn3, Pn2, Pn1 ]
        else: pts = [ P0, P1, P2, *self.interior, Pn3, Pn2, Pn1 ]

        return torch.stack(pts, dim=0)        
    
    def _bernstein(self, u, degree):
        i = torch.arange(degree + 1, device=u.device)

        binom = torch.exp( torch.lgamma(torch.tensor(degree + 1.0, device=u.device)) - torch.lgamma(i + 1.0) - torch.lgamma(degree - i + 1.0) )

        return binom * (u[..., None] ** i) * ((1.0 - u[..., None]) ** (degree - i))

    
    def position(self, t):
        P = self.control_points()     
        B = self._bernstein(t, self.n_ctrl - 1)
        return B @ P

    def velocity(self, t):
        P = self.control_points()
        dP = (self.n_ctrl - 1) * (P[1:] - P[:-1])
        B = self._bernstein(t, self.n_ctrl - 2)
        return B @ dP

    def accel(self, t):
        P = self.control_points()
        ddP = ( (self.n_ctrl - 1) * (self.n_ctrl - 2) * (P[2:] - 2 * P[1:-1] + P[:-2]) )
        B = self._bernstein(t, self.n_ctrl - 3)
        return B @ ddP

class BSplineCurve(ArclenParameterize):
    def __init__(self, n_ctrl: int, P_T: torch.Tensor, degree: int = 3):
        assert degree == 3, "This implementation assumes cubic B-splines"
        assert n_ctrl >= 6

        super().__init__(interval=[0.0, 1.0])

        self.n_ctrl = n_ctrl
        self.degree = degree
        self.dim = P_T.shape[-1]

        self.register_buffer("P_T", P_T)

        self.C1 = nn.Parameter(torch.randn(self.dim))
        self.Cn1 = nn.Parameter(torch.randn(self.dim))

        n_interior = n_ctrl - 6
        if n_interior > 0: self.interior = nn.Parameter(torch.randn(n_interior, self.dim))
        else: self.interior = None

        self.register_buffer("knots", self._make_knots())

    def control_points(self):
        C0 = torch.zeros_like(self.P_T)
        C1 = self.C1
        C2 = 2.0 * C1

        Cn1 = self.Cn1
        Cn2 = 2.0 * Cn1 - self.P_T
        Cn = self.P_T

        if self.interior is None: pts = [C0, C1, C2, Cn2, Cn1, Cn]
        else: pts = [C0, C1, C2, *self.interior, Cn2, Cn1, Cn]

        return torch.stack(pts, dim=0)

    def _make_knots(self):
        p = self.degree
        n = self.n_ctrl - 1
        m = n + p + 1

        knots = torch.zeros(m + 1)
        knots[p : m - p + 1] = torch.linspace(0.0, 1.0, m - 2 * p + 1)
        knots[m - p + 1 :] = 1.0
        return knots

    def _basis(self, t, i, p, knots):
        if p == 0:
            return ((knots[i] <= t) & (t < knots[i + 1])).to(t.dtype)

        denom1 = knots[i + p] - knots[i]
        denom2 = knots[i + p + 1] - knots[i + 1]

        term1 = torch.where(
            denom1 > 0,
            (t - knots[i]) / denom1 * self._basis(t, i, p - 1, knots),
            torch.zeros_like(t),
        )

        term2 = torch.where(
            denom2 > 0,
            (knots[i + p + 1] - t) / denom2 * self._basis(t, i + 1, p - 1, knots),
            torch.zeros_like(t),
        )

        return term1 + term2
    
    def _basis_matrix(self, t, p, knots, n_basis: int):
        B = [self._basis(t, i, p, knots) for i in range(n_basis)]
        return torch.stack(B, dim=-1)

    def position(self, t):
        C = self.control_points()                     
        B = self._basis_matrix(t, self.degree, self.knots, self.n_ctrl)  
        return B @ C                                  

    def velocity(self, t):
        C = self.control_points()
        p = self.degree
        U = self.knots

        denom = (U[p+1:p+self.n_ctrl] - U[1:self.n_ctrl]).unsqueeze(-1)  
        dC = p * (C[1:] - C[:-1]) / denom                                 

        U1 = U[1:-1]  
        B = self._basis_matrix(t, p-1, U1, self.n_ctrl - 1)               
        return B @ dC

    def accel(self, t):
        C = self.control_points()
        p = self.degree
        U = self.knots

        denom1 = (U[p+1:p+self.n_ctrl] - U[1:self.n_ctrl]).unsqueeze(-1)
        dC = p * (C[1:] - C[:-1]) / denom1                                 

        denom2 = (U[p+1:p+self.n_ctrl-1] - U[2:self.n_ctrl]).unsqueeze(-1)
        ddC = (p-1) * (dC[1:] - dC[:-1]) / denom2                          

        U2 = U[2:-2]  
        B = self._basis_matrix(t, p-2, U2, self.n_ctrl - 2)                
        return B @ ddC

def pinv_and_nullspace(M: torch.Tensor, tol: float = 1e-10):
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)

    # Robust threshold
    if S.numel() == 0:
        # Degenerate: M is empty
        n = M.shape[1]
        M_pinv = torch.zeros((n, M.shape[0]), device=M.device, dtype=M.dtype)
        N = torch.eye(n, device=M.device, dtype=M.dtype)
        return M_pinv, N

    Smax = S.max()
    thresh = tol * Smax
    r = int((S > thresh).sum().item())  # numerical rank

    m, n = M.shape
    # Build pseudoinverse: V[:, :r] diag(1/S[:r]) U[:, :r]^T
    if r == 0:
        M_pinv = torch.zeros((n, m), device=M.device, dtype=M.dtype)
        N = Vh.transpose(-2, -1)  # all of R^n is nullspace if M ~ 0
        return M_pinv, N

    V = Vh.transpose(-2, -1)  # (n, n)
    U_r = U[:, :r]            # (m, r)
    V_r = V[:, :r]            # (n, r)
    S_r_inv = (1.0 / S[:r]).to(M.dtype)  # (r,)

    # (n, r) @ (r, r) @ (r, m) -> (n, m)
    M_pinv = V_r @ torch.diag(S_r_inv) @ U_r.transpose(-2, -1)

    # Nullspace basis: columns of V corresponding to zero singular values
    if r < n:
        N = V[:, r:]  # (n, n-r)
    else:
        N = torch.zeros((n, 0), device=M.device, dtype=M.dtype)

    return M_pinv, N


class FourierCurve(ArclenParameterize):
    def __init__(
        self,
        final_T: float,
        frequencies: torch.Tensor,
        *fixed_pts: tuple[float, torch.Tensor, int],
        cache_X: bool = True,
        tol: float = 1e-10,
    ):
        super().__init__( interval=[0, final_T] )

        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1D tensor of shape (N,)")

        self.T = float(final_T)
        self.register_buffer("freq", frequencies.detach().clone().to(dtype=torch.get_default_dtype()))

        if len(fixed_pts) == 0:
            raise ValueError("Provide at least one fixed point constraint.")

        times = torch.tensor([float(t) for (t, _, _) in fixed_pts], dtype=self.freq.dtype, device=self.freq.device)
        derivs = torch.tensor([int(k) for (_, _, k) in fixed_pts], dtype=torch.long, device=self.freq.device)
        
        R = torch.stack([torch.as_tensor(v, device=self.freq.device, dtype=self.freq.dtype) for (_, v, _) in fixed_pts], dim=0)
        if R.ndim != 2 or R.shape[1] != 3:
            raise ValueError("Each fixed point value must be a 3D vector (shape (3,)).")

        self.register_buffer("times", times)
        self.register_buffer("derivs", derivs)

        M = self._build_M(times=self.times, derivs=self.derivs, freq=self.freq)  # (K, 2N)

        M_pinv, null_M = pinv_and_nullspace(M, tol=tol)  # (2N,K), (2N, dim_null)

        self.register_buffer("M", M)
        self.register_buffer("M_pinv", M_pinv)
        self.register_buffer("null_M", null_M)
        self.register_buffer("particular_sol", M_pinv @ R)  # (2N, 3)

        dim_null = null_M.shape[1]
        self.Z = nn.Parameter(torch.randn(dim_null, 3, device=self.freq.device, dtype=self.freq.dtype) * 0.01)

        self._cache_enabled = bool(cache_X)
        self.register_buffer("_X_cache", torch.empty((0, 0), device=self.freq.device, dtype=self.freq.dtype))
        self._X_cache_valid = False
        self._Z_version_at_cache = -1  

    @staticmethod
    def _build_M(times: torch.Tensor, derivs: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Build M so that (K, 2N) @ (2N,3) matches constraints on r^(k)(t):
          rows correspond to constraints (t_i, k_i)
          columns correspond to [a_1..a_N, b_1..b_N] for cos/sin blocks
        """
        K = times.shape[0]
        N = freq.shape[0]

        t = times[:, None]         # (K,1)
        w = freq[None, :]          # (1,N)
        phase = w * t              # (K,N)

        cos = torch.cos(phase)     # (K,N)
        sin = torch.sin(phase)     # (K,N)

        M_cos = torch.zeros((K, N), device=freq.device, dtype=freq.dtype)
        M_sin = torch.zeros((K, N), device=freq.device, dtype=freq.dtype)

        # Vectorized per-row via masks
        d = derivs.to(torch.long)              # (K,)
        dmod = (d % 4).to(torch.long)          # (K,)
        w_pow = (w ** d[:, None].to(freq.dtype))  # (K,N)

        # dmod == 0:  cos ->  cos,   sin ->  sin
        m0 = (dmod == 0)
        if m0.any():
            M_cos[m0] = (w_pow[m0] * cos[m0])
            M_sin[m0] = (w_pow[m0] * sin[m0])

        # dmod == 1:  cos -> -sin,   sin ->  cos
        m1 = (dmod == 1)
        if m1.any():
            M_cos[m1] = (-w_pow[m1] * sin[m1])
            M_sin[m1] = ( w_pow[m1] * cos[m1])

        # dmod == 2:  cos -> -cos,   sin -> -sin
        m2 = (dmod == 2)
        if m2.any():
            M_cos[m2] = (-w_pow[m2] * cos[m2])
            M_sin[m2] = (-w_pow[m2] * sin[m2])

        # dmod == 3:  cos ->  sin,   sin -> -cos
        m3 = (dmod == 3)
        if m3.any():
            M_cos[m3] = ( w_pow[m3] * sin[m3])
            M_sin[m3] = (-w_pow[m3] * cos[m3])

        return torch.cat([M_cos, M_sin], dim=1)  # (K, 2N)

    # ---- Cache controls ----
    def invalidate_cache(self):
        self._X_cache_valid = False

    def enable_cache(self):
        self._cache_enabled = True
        self.invalidate_cache()

    def disable_cache(self):
        self._cache_enabled = False
        self.invalidate_cache()

    @property
    def X(self) -> torch.Tensor:
        """
        Returns (2N,3) coefficient matrix X = [a; b].
        Cached if enabled; automatically invalidates if Z changed.
        """
        if not self._cache_enabled:
            return self.particular_sol + (self.null_M @ self.Z)

        zver = getattr(self.Z, "_version", None)
        if (not self._X_cache_valid) or (zver is not None and zver != self._Z_version_at_cache):
            X = self.particular_sol + (self.null_M @ self.Z)
            self._X_cache = X
            self._X_cache_valid = True
            self._Z_version_at_cache = -1 if zver is None else zver
        return self._X_cache

    def position(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.freq
        X = self.X

        t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
        scalar_input = (t.ndim == 0)
        if scalar_input:
            t = t.unsqueeze(0)  # (1,)

        phase = t[:, None] * freq[None, :]  # (B,N)
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]   # a_n (N,3)
        Xs = X[N:, :]   # b_n (N,3)

        r = cos @ Xc + sin @ Xs  # (B,3)
        return r.squeeze(0) if scalar_input else r

    def velocity(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.freq
        X = self.X

        t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
        scalar_input = (t.ndim == 0)
        if scalar_input:
            t = t.unsqueeze(0)

        phase = t[:, None] * freq[None, :]
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]
        Xs = X[N:, :]

        w = freq[None, :]  # (1,N)
        v = (-w * sin) @ Xc + (w * cos) @ Xs
        return v.squeeze(0) if scalar_input else v

    def accel(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.freq
        X = self.X

        t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
        scalar_input = (t.ndim == 0)
        if scalar_input:
            t = t.unsqueeze(0)

        phase = t[:, None] * freq[None, :]
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]
        Xs = X[N:, :]

        w2 = (freq[None, :] ** 2)
        a = (-w2 * cos) @ Xc + (-w2 * sin) @ Xs
        return a.squeeze(0) if scalar_input else a
