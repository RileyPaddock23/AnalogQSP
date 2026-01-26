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

def pinv_and_nullspace(M: torch.Tensor, tol: float = 1e-10):
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)

    if S.numel() == 0:
        n = M.shape[1]
        M_pinv = torch.zeros((n, M.shape[0]), device=M.device, dtype=M.dtype)
        N = torch.eye(n, device=M.device, dtype=M.dtype)
        return M_pinv, N

    Smax = S.max()
    thresh = tol * Smax
    r = int((S > thresh).sum().item())  

    m, n = M.shape
    
    if r == 0:
        M_pinv = torch.zeros((n, m), device=M.device, dtype=M.dtype)
        N = Vh.transpose(-2, -1)  
        return M_pinv, N

    V = Vh.transpose(-2, -1)  
    U_r = U[:, :r]            
    V_r = V[:, :r]            
    S_r_inv = (1.0 / S[:r]).to(M.dtype)  

    M_pinv = V_r @ torch.diag(S_r_inv) @ U_r.transpose(-2, -1)

    if r < n: N = V[:, r:]  
    else: N = torch.zeros((n, 0), device=M.device, dtype=M.dtype)

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

        if frequencies.ndim != 1: raise ValueError("frequencies must be a 1D tensor of shape (N,)")

        self.T = float(final_T)
        self.register_buffer("freq", frequencies.detach().clone().to(dtype=torch.get_default_dtype()))

        if len(fixed_pts) == 0: raise ValueError("Provide at least one fixed point constraint.")

        times = torch.tensor([float(t) for (t, _, _) in fixed_pts], dtype=self.freq.dtype, device=self.freq.device)
        derivs = torch.tensor([int(k) for (_, _, k) in fixed_pts], dtype=torch.long, device=self.freq.device)
        
        R = torch.stack([torch.as_tensor(v, device=self.freq.device, dtype=self.freq.dtype) for (_, v, _) in fixed_pts], dim=0)
        if R.ndim != 2 or R.shape[1] != 3: raise ValueError("Each fixed point value must be a 3D vector (shape (3,)).")

        self.register_buffer("times", times)
        self.register_buffer("derivs", derivs)

        M = self._build_M(times=self.times, derivs=self.derivs, freq=self.freq)  

        M_pinv, null_M = pinv_and_nullspace(M, tol=tol)  

        self.register_buffer("M", M)
        self.register_buffer("M_pinv", M_pinv)
        self.register_buffer("null_M", null_M)
        self.register_buffer("particular_sol", M_pinv @ R)  

        dim_null = null_M.shape[1]
        self.Z = nn.Parameter(torch.randn(dim_null, 3, device=self.freq.device, dtype=self.freq.dtype) * 0.01)

        self._cache_enabled = bool(cache_X)
        self.register_buffer("_X_cache", torch.empty((0, 0), device=self.freq.device, dtype=self.freq.dtype))
        self._X_cache_valid = False
        self._Z_version_at_cache = -1  

    @staticmethod
    def _build_M(times: torch.Tensor, derivs: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        K = times.shape[0]
        N = freq.shape[0]

        t = times[:, None]         
        w = freq[None, :]          
        phase = w * t              

        cos = torch.cos(phase)     
        sin = torch.sin(phase)     

        M_cos = torch.zeros((K, N), device=freq.device, dtype=freq.dtype)
        M_sin = torch.zeros((K, N), device=freq.device, dtype=freq.dtype)

        d = derivs.to(torch.long)              
        dmod = (d % 4).to(torch.long)          
        w_pow = (w ** d[:, None].to(freq.dtype))  

        m0 = (dmod == 0)
        if m0.any():
            M_cos[m0] = (w_pow[m0] * cos[m0])
            M_sin[m0] = (w_pow[m0] * sin[m0])

        m1 = (dmod == 1)
        if m1.any():
            M_cos[m1] = (-w_pow[m1] * sin[m1])
            M_sin[m1] = ( w_pow[m1] * cos[m1])

        m2 = (dmod == 2)
        if m2.any():
            M_cos[m2] = (-w_pow[m2] * cos[m2])
            M_sin[m2] = (-w_pow[m2] * sin[m2])

        m3 = (dmod == 3)
        if m3.any():
            M_cos[m3] = ( w_pow[m3] * sin[m3])
            M_sin[m3] = (-w_pow[m3] * cos[m3])

        return torch.cat([M_cos, M_sin], dim=1)  

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
        if not self._cache_enabled: return self.particular_sol + (self.null_M @ self.Z)

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
        if scalar_input: t = t.unsqueeze(0)  

        phase = t[:, None] * freq[None, :]  
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]   
        Xs = X[N:, :]   

        r = cos @ Xc + sin @ Xs  
        return r.squeeze(0) if scalar_input else r

    def velocity(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.freq
        X = self.X

        t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
        scalar_input = (t.ndim == 0)
        if scalar_input: t = t.unsqueeze(0)

        phase = t[:, None] * freq[None, :]
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]
        Xs = X[N:, :]

        w = freq[None, :]  
        v = (-w * sin) @ Xc + (w * cos) @ Xs
        return v.squeeze(0) if scalar_input else v

    def accel(self, t: torch.Tensor) -> torch.Tensor:
        freq = self.freq
        X = self.X

        t = torch.as_tensor(t, device=X.device, dtype=X.dtype)
        scalar_input = (t.ndim == 0)
        if scalar_input: t = t.unsqueeze(0)

        phase = t[:, None] * freq[None, :]
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        N = freq.shape[0]
        Xc = X[:N, :]
        Xs = X[N:, :]

        w2 = (freq[None, :] ** 2)
        a = (-w2 * cos) @ Xc + (-w2 * sin) @ Xs
        return a.squeeze(0) if scalar_input else a
