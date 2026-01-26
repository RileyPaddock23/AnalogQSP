import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchquad.integration.monte_carlo import MonteCarlo
from torchquad.integration.simpson import Simpson
import matplotlib.pyplot as plt
from tqdm import trange

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

    def magnus_first_order(self):
        return self(self.interval[1]) - self(self.interval[0])

    def magnus_second_order(self, integrator_cls=Simpson, samples=8192):
        t0, t1 = self.interval

        def f(t): return torch.linalg.cross(self.curve_d1(t), self.curve(t))

        integrator = integrator_cls()
        return integrator.integrate( f, dim=1, N=samples, integration_domain=[[t0, t1]] )
    
    def magnus_third_order(self, samples=8192):
        t0, t1 = self.interval

        t = torch.linspace(t0, t1, samples)

        r = self.curve(t)
        rp = self.curve_d1(t)

        inner = torch.linalg.cross(rp, r)

        dt = (t1 - t0) / (samples - 1)
        A = torch.cumsum(inner, dim=0) * dt

        outer = torch.linalg.cross(A, rp)

        term1 = torch.sum(outer, dim=0) * dt

        corr = torch.linalg.cross(r, torch.linalg.cross(r, rp))
        term2 = torch.sum(corr, dim=0) * dt

        return term1 - (2.0 / 3.0) * term2
    
    def magnus_fourth_order(self, samples=8192, return_curve=False):
        t0, t1 = self.interval
        t = torch.linspace(t0, t1, samples)
        dt = (t1 - t0) / (samples - 1)

        r = self.curve(t)      
        T = self.curve_d1(t)   

        u = _cumtrapz(torch.linalg.cross(T, r), dt)      

        w = _cumtrapz(torch.linalg.cross(T, u), dt)        

        I1 = 2.0 * _cumtrapz(torch.linalg.cross(T, w), dt)  
        Mx = _cumtrapz(T * u[:, 0:1], dt)
        My = _cumtrapz(T * u[:, 1:2], dt)
        Mz = _cumtrapz(T * u[:, 2:3], dt)

        S = _cumtrapz(torch.sum(T * u, dim=1, keepdim=True), dt)
        G = _cumtrapz(torch.linalg.cross(torch.linalg.cross(r, T), u), dt)

        first_term = r[:, 0:1] * Mx + r[:, 1:2] * My + r[:, 2:3] * Mz  
        second_term = r * S                                            
        I2 = (first_term - second_term) - G                             

        K4 = I1 + I2                                                    

        if return_curve: return t, K4
        return K4[-1]

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
        return torch.tensor([ 
            self.magnus_first_order().reshape(-1)[2], 
            self.magnus_second_order().reshape(-1)[2], 
            self.magnus_third_order()[2],
            self.magnus_fourth_order()[2]
        ])

    def polynomial_coeffs_y(self):
        return torch.tensor([ 
            self.magnus_first_order().reshape(-1)[1], 
            self.magnus_second_order().reshape(-1)[1], 
            self.magnus_third_order()[1],
            self.magnus_fourth_order()[1]
        ])

    def polynomial_coeffs_x(self):
        return torch.tensor([ 
            self.magnus_first_order().reshape(-1)[0], 
            self.magnus_second_order().reshape(-1)[0], 
            self.magnus_third_order()[0],
            self.magnus_fourth_order()[0]
        ])

    
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
        lambda curve, expected: torch.linalg.norm(curve.magnus_second_order() - expected),
        lambda curve, expected: torch.linalg.norm(curve.magnus_third_order() - expected),
        lambda curve, expected: torch.linalg.norm(curve.magnus_fourth_order() - expected)
    ]

    def __init__(self, curve: LearnableCurve, expected_magnus_terms : torch.Tensor, weights : torch.Tensor):
        assert len(expected_magnus_terms.shape) == 2 and expected_magnus_terms.shape[1] == 3 and expected_magnus_terms.shape[0] < 5
        assert weights.shape[0] == expected_magnus_terms.shape[0]

        self.curve = curve
        self.exp_mag_terms = expected_magnus_terms
        self.weights = weights
    
    def cost(self, _):
        terms = []
        terms.append(self.curve.soft_max_curvature().expand(3))
        terms.append(self.curve.magnus_second_order().squeeze(0))
        terms.append(self.curve.magnus_third_order().squeeze(0))
        terms.append(self.curve.magnus_fourth_order().squeeze(0))

        terms = torch.stack(terms[:len(self.exp_mag_terms)])

        diffs = torch.linalg.norm(terms - self.exp_mag_terms, dim=1)
        return torch.dot(self.weights, diffs)

    def optimize(self, lr=1e-3, steps=10_000, callback=None):

        self.curve.optimize(  lambda _: self.cost(_) , lr=lr, steps=steps, callback=callback )
    
    def poly_z(self): return [float(value) for value in self.curve.polynomial_coeffs_z()]
    def poly_y(self): return [float(value) for value in self.curve.polynomial_coeffs_y()]
    def poly_x(self): return [float(value) for value in self.curve.polynomial_coeffs_x()]