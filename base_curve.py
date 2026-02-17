import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
from tqdm import trange
from magnus import MagnusODE

from abc import ABCMeta, abstractmethod
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple
def _cumtrapz(y, dt):
    return torch.cat(
        [torch.zeros_like(y[:1]), torch.cumsum(0.5 * (y[1:] + y[:-1]) * dt, dim=0)],
        dim=0,
    )

class LearnableCurve(nn.Module, metaclass=ABCMeta):
    def __init__(self, interval: tuple[float, float]):
        super().__init__()
        self.interval = interval

    @abstractmethod
    def curve(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def curve_d1(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def curve_d2(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def _as_tensor(self, t) -> torch.Tensor:
        if isinstance(t, torch.Tensor): return t
        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        return torch.tensor(t, dtype=torch.float32, device=device)


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self._as_tensor(t)

        if t.dim() == 0: t = t.unsqueeze(0)
        out = self.curve(t)
        if out.shape[-1] != 3: raise ValueError("curve output should be R^3, i.e. last dim should be 3")
        return out

    def curvature(self, t: torch.Tensor, eps = 1e-8) -> torch.Tensor:
        v = self.curve_d1(t)
        a = self.curve_d2(t)

        cross = torch.linalg.cross(v, a, dim=-1)
        num = torch.linalg.norm(cross, dim=-1)

        denom = torch.linalg.norm(v, dim=-1) ** 3

        return num / (denom + eps)

    def max_curvature(self, samples = 4096):
        return self.curvature( torch.linspace(self.interval[0], self.interval[1], samples) ).max()

    def soft_max_curvature(self, samples = 4096, strength = 32):
        k = self.curvature( torch.rand( samples ) )
        return torch.logsumexp(k * strength, dim=0) / strength

    def magnus(self, K):
        #Build ODE system to find K Magnus terms at time T
        ode = MagnusODE(self, K)
        tangent0 = self.curve_d1(torch.as_tensor(0.0))
        if tangent0.ndim > 1:
            tangent0 = tangent0.squeeze(0)

        dtype = torch.complex64 if not torch.is_complex(tangent0) else tangent0.dtype
        X0 = torch.zeros(3 * K, dtype=dtype, device=tangent0.device)

        t = torch.linspace(0.0, self.interval[1], steps=2, dtype=tangent0.real.dtype, device=tangent0.device)

        #We only care about evaluating at the final time T which by FToC means
        #we only need to evaluate each Omega_n at time 0 and time T
        XT = odeint_adjoint(ode, X0, t)[-1]

        self.recent_magnus_terms = XT.view(K, 3)
        return XT.view(K, 3)

    def make_cost_fn(self, *terms: tuple[float, Callable[['LearnableCurve', tuple], torch.Tensor]] ):

        def cost_fn(curve) -> torch.Tensor:
            total = 0.0

            for weight, fn in terms: total += weight * fn(curve)

            return total

        return cost_fn

    def optimize(
        self,
        cost_fn : Callable[ ['LearnableCurve', tuple], torch.Tensor ],
        lr: float = 1e-3,
        steps: int = 1000,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        callback=None,
    ):
    
        if optimizer_kwargs is None: optimizer_kwargs = {}
        optimizer = optimizer_cls(self.parameters(), lr=lr, **optimizer_kwargs)

        for step in range(steps):
            optimizer.zero_grad()
            loss = cost_fn(self)
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()

            if callback is not None: callback(step, loss_value, self)

    # could make more efficient, too lazy

    def polynomial_coeffs_z(self):
        return self.recent_magnus_terms[:,2]

    def polynomial_coeffs_y(self):
        return self.recent_magnus_terms[:,1]

    def polynomial_coeffs_x(self):
        return self.recent_magnus_terms[:,0]


    def plot_position(self, samples: int = 1024, ax=None, show=True):
        t0, t1 = self.interval
        t = torch.linspace(t0, t1, samples, device=next(self.parameters()).device)

        with torch.no_grad(): p = self.curve(t).cpu()

        x, y, z = p[:, 0], p[:, 1], p[:, 2]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.plot(x, y, z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Curve position")

        if show: plt.show()

        return ax

    def plot_curvature(self, samples: int = 1024, ax=None, show=True):
        t0, t1 = self.interval
        t = torch.linspace(t0, t1, samples, device=next(self.parameters()).device)

        with torch.no_grad(): k = self.curvature(t).cpu()

        if ax is None: fig, ax = plt.subplots()

        ax.plot(t.cpu(), k)
        ax.set_xlabel("t")
        ax.set_ylabel("curvature")
        ax.set_title("Curvature along curve")

        if show: plt.show()

        return ax



class ArclenParameterize(LearnableCurve, metaclass=ABCMeta):

    @abstractmethod
    def position(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def velocity(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def accel(self, t: torch.Tensor) -> torch.Tensor:
        pass

    def _build_arclen_table(self, samples : int | None = None):
        if samples is None: samples = self._arc_samples

        t0, t1 = self.t_interval

        param = next(self.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        u = torch.linspace(t0, t1, samples, device=device)

        v = self.velocity(u)
        speed = torch.linalg.norm(v, dim=-1)

        du = (t1 - t0) / (samples - 1)
        s = torch.cumsum(speed, dim=0) * du
        s = torch.cat([torch.zeros(1, device=s.device), s[:-1]])
        s = s / s[-1]

        self._arc_u = u.detach()
        self._arc_s = s.detach()
        self._arc_dirty = False

    def __init__(self, interval: tuple[float, float], samples = 4096):
        self._arc_u = None
        self._arc_s = None
        self._arc_dirty = True
        self.t_interval = interval
        self._arc_samples = samples
        super().__init__(interval=[0, 1])

    def _u_from_t(self, t: torch.Tensor) -> torch.Tensor:
        t = self._as_tensor(t)

        if self._arc_dirty or self._arc_u is None: self._build_arclen_table()

        s = self._arc_s
        u = self._arc_u

        t = t.clamp(0.0, 1.0)

        idx = torch.searchsorted(s, t) - 1
        idx = idx.clamp(0, len(s) - 2)

        s0, s1 = s[idx], s[idx + 1]
        u0, u1 = u[idx], u[idx + 1]

        w = (t - s0) / (s1 - s0 + 1e-8)
        return u0 + w * (u1 - u0)

    def _ensure_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1: return x.unsqueeze(0)
        return x


    def curve(self, t):
        t = self._as_tensor(t)
        u = self._u_from_t(t)
        out = self.position(u)
        return self._ensure_batch(out)

    def curve_d1(self, t):
        t = self._as_tensor(t)
        u = self._u_from_t(t)
        v = self.velocity(u)
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
        return self._ensure_batch(v)

    def curve_d2(self, t):
        t = self._as_tensor(t)
        u = self._u_from_t(t)

        v = self.velocity(u)
        a = self.accel(u)

        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        v_hat = v / speed

        proj = (a * v_hat).sum(dim=-1, keepdim=True) * v_hat
        a_perp = a - proj

        return self._ensure_batch(a_perp / (speed ** 2))

    def curvature(self, t):
        return torch.linalg.norm(self.curve_d2(t), dim=-1)

    def invalidate_arclen(self): self._arc_dirty = True


class CurveTester:
    fns = [
        lambda curve, _: curve.soft_max_curvature(),
        lambda curve, K,expected: torch.sum((curve.magnus(K) - expected)**2)
    ]

    def __init__(self, curve: LearnableCurve, expected_magnus_terms : torch.Tensor, weights : torch.Tensor):
        assert weights.shape[0] == expected_magnus_terms.shape[0]

        self.curve = curve
        self.exp_mag_terms = expected_magnus_terms
        self.K = expected_magnus_terms.shape[0]
        self.weights = weights

    def cost(self, _):
        # curvature = self.soft_max_curvature().expand(1)
        magnus_terms = self.curve.magnus(self.K)

        diffs = (magnus_terms - self.exp_mag_terms)**2
        # terms = torch.concat((curvature, diffs))
        # R = compute_R_terminal(curve.curve_d1, K)

        # loss = torch.sum((R - target_R)**2)
        return torch.sum(diffs)

    def optimize(self, lr=1e-3, steps=10_000, callback=None):

        self.curve.optimize(lambda _: self.cost(_) , lr=lr, steps=steps, callback=callback)

    def poly_z(self): return [float(value) for value in self.curve.polynomial_coeffs_z()]
    def poly_y(self): return [float(value) for value in self.curve.polynomial_coeffs_y()]
    def poly_x(self): return [float(value) for value in self.curve.polynomial_coeffs_x()]
