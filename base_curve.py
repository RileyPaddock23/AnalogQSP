import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

from abc import ABCMeta, abstractmethod
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0: t = t.unsqueeze(0)
        out = self.position(t)
        if out.shape[-1] != 3: raise ValueError("curve output should be R^3, i.e. last dim should be 3")
        return out
    
    def tangent(self, t: torch.Tensor) -> torch.Tensor:
        velocity = self.velocity(t)
        return velocity / torch.linalg.norm(velocity, dim=-1, keepdim=True)
    
    def curvature(self, t: torch.Tensor, eps = 1e-8) -> torch.Tensor:
        v = self.curve_d1(t)
        a = self.curve_d2(t)

        cross = torch.linalg.cross(v, a, dim=-1)
        num = torch.linalg.norm(cross, dim=-1)

        denom = torch.linalg.norm(v, dim=-1) ** 3

        return num / (denom + eps)
    
    def max_curvature(self, samples = 4096):
        return self.curvature( torch.linspace(self.interval[0], self.interval[1], samples) ).max()
    
    def soft_max_curvature(self, samples = 4096, strength = 16):
        k = self.curvature(samples)
        return torch.logsumexp(k * strength, dim=0) / strength
    
    def make_cost_fn(self, *terms: tuple[float, Callable, tuple] ):

        def cost_fn(t: torch.Tensor) -> torch.Tensor:
            total = 0.0
            
            for weight, fn, args in terms:
                out = fn(self, t, *args)
                
                total += weight * out

            return total

        return cost_fn
    
    def optimize(
        self,
        cost_fn,
        t: torch.Tensor,
        lr: float = 1e-3,
        steps: int = 1000,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        callback=None,
    ):
        if optimizer_kwargs is None: optimizer_kwargs = {}

        optimizer = optimizer_cls(self.parameters(), lr=lr, **optimizer_kwargs)

        losses = []

        for step in range(steps):
            optimizer.zero_grad()

            loss = cost_fn(t)
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()
            losses.append(loss_value)

            if callback is not None: callback(step, loss, self)

        return losses
            

class MagnusExpression:
    @dataclass
    class Config:
        steps: int = 4096
        method: str = "dopri5"
        rtol: float = 1e-5
        atol: float = 1e-7
        use_dense_cache: bool = True

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or MagnusExpression.Config()
        self.order: int = 0

        self.pi: List[Callable[[LearnableCurve, float], torch.Tensor]] = []
        self.s: List[List[Callable[[LearnableCurve, float], torch.Tensor]]] = []

        self._B: List[float] = [1.0, -0.5]

        self._int_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _ensure_bernoulli(self, n: int) -> None:
        while len(self._B) <= n:
            m = len(self._B)
            s = 0.0
            for k in range(m): s += math.comb(m + 1, k) * self._B[k]
            self._B.append(-s / (m + 1))

    def _Bnum(self, j: int) -> float:
        self._ensure_bernoulli(j)
        return float(self._B[j])

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor: return x if x.is_complex() else x.to(torch.cfloat)

    def _zeros_like(self, curve: LearnableCurve, t: float) -> torch.Tensor: return torch.zeros_like(self._to_complex(curve(t)))

    @staticmethod
    def _interp_linear(ts: torch.Tensor, ys: torch.Tensor, t: float) -> torch.Tensor:
        t = float(t)
        if t <= 0.0: return ys[0]

        tt = torch.tensor([t], device=ts.device, dtype=ts.dtype)
        idx = int(torch.searchsorted(ts, tt).item()) - 1
        idx = max(0, min(idx, ts.numel() - 2))

        t0 = float(ts[idx].item())
        t1 = float(ts[idx + 1].item())
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1.0 - w) * ys[idx] + w * ys[idx + 1]

    def _integrate_cached( self, tag: str, f: Callable[[LearnableCurve, float], torch.Tensor], curve: LearnableCurve, t: float) -> torch.Tensor:
        t = float(t)
        if t <= 0.0: return self._zeros_like(curve, 0.0)

        key = (id(curve), tag)
        rebuild = (not self.cfg.use_dense_cache) or (key not in self._int_cache)

        if not rebuild and self.cfg.use_dense_cache:
            ts_cached, _ = self._int_cache[key]
            if t > float(ts_cached[-1].item()) + 1e-12: rebuild = True

        if rebuild:
            y0 = self._zeros_like(curve, 0.0)
            device = y0.device

            ts = torch.linspace(0.0, t, self.cfg.steps, device=device, dtype=torch.float32)

            def rhs(tt: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return self._to_complex(f(curve, float(tt.item())))

            ys = odeint_adjoint(
                rhs,
                y0,
                ts,
                method=self.cfg.method,
                rtol=self.cfg.rtol,
                atol=self.cfg.atol,
            )

            if self.cfg.use_dense_cache: self._int_cache[key] = (ts.detach(), ys.detach())
            else: return ys[-1]

        ts, ys = self._int_cache[key]
        return self._interp_linear(ts, ys, t)

    def generate(self, order: int) -> None:
        if order < 0: raise ValueError("order must be >= 0")
        for n in range(self.order + 1, order + 1): self._generate_order(n)

    def __getitem__(self, n: int) -> Callable[[LearnableCurve, float], torch.Tensor]:
        if n < 1: raise RuntimeError("orders less than 1 do not exist")
        if n > self.order: self.generate(n)
        return self.pi[n - 1]

    def _generate_order(self, n: int) -> None:
        while len(self.pi) < n: self.pi.append(lambda _c, _t: None)  
        while len(self.s) < n: self.s.append([])

        if n == 1:
            def pi_1(curve: LearnableCurve, t: float) -> torch.Tensor: return self._to_complex(curve(t) - curve(0.0))
            self.pi[0] = pi_1
            self.order = 1
            return

        pi_prev = self.pi[n - 2]

        def S_n_1(curve: LearnableCurve, t: float) -> torch.Tensor:
            a = self._to_complex(pi_prev(curve, t))
            b = self._to_complex(curve.tangent(t))
            return 2j * torch.cross(a, b)

        Sn_list: List[Callable[[LearnableCurve, float], torch.Tensor]] = [S_n_1]

        for j in range(2, n):
            def make_S_n_j(j_local: int):
                def S_n_j(curve: LearnableCurve, t: float) -> torch.Tensor:
                    acc = self._zeros_like(curve, t)
                    for m in range(1, n - j_local + 1):
                        Pi_m = self.pi[m - 1]
                        S_nm_prev = self.s[n - m - 1][j_local - 2]
                        acc = acc + torch.cross( self._to_complex(Pi_m(curve, t)), self._to_complex(S_nm_prev(curve, t)))
                    return 2j * acc
                return S_n_j

            Sn_list.append(make_S_n_j(j))

        self.s[n - 1] = Sn_list

        def pi_n(curve: LearnableCurve, t: float) -> torch.Tensor:
            total = self._zeros_like(curve, t)

            for j in range(1, n):
                if j > 1 and (j % 2 == 1): continue

                Bj = self._Bnum(j)
                if Bj == 0.0: continue

                Sj = self.s[n - 1][j - 1]
                integral = self._integrate_cached(tag=f"S(n={n},j={j})", f=Sj, curve=curve, t=t)
                total = total + (Bj / math.factorial(j)) * self._to_complex(integral)

            return total

        self.pi[n - 1] = pi_n
        self.order = max(self.order, n)
