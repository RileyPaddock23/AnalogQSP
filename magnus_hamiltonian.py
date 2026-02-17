"""Numerical Magnus-term solver for time-dependent Hamiltonians.

This module provides a standalone solver that computes the first ``K`` terms
of the Magnus expansion at a terminal time ``T`` for

    dU/dt = A(t) U,

where ``A(t)`` is usually ``-1j * H(t)`` for a Hermitian Hamiltonian ``H(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import bernoulli, factorial


ArrayLike = np.ndarray


@dataclass
class MagnusResult:
    """Container for terminal Magnus terms."""

    terms: np.ndarray  # shape: (K, d, d)
    t_final: float


class HamiltonianMagnusODE:
    """ODE system for recursively evolving Magnus terms.

    The recursion implemented is

    d/dt Ω_n = Σ_{j=1}^{n-1} (B_j / j!)
              Σ_{i_1+...+i_j=n-1} ad_{Ω_{i_1}} ... ad_{Ω_{i_j}} A(t)

    with Ω_1'(t) = A(t), where ``ad_X(Y) = [X, Y] = XY - YX``.
    """

    def __init__(self, A: Callable[[float], ArrayLike], K: int, dim: int):
        if K < 1:
            raise ValueError("K must be >= 1")
        self.A = A
        self.K = int(K)
        self.dim = int(dim)
        self.bernoulli_coeffs = bernoulli(self.K)

    def _compositions(self, n: int, m: int) -> Iterable[tuple[int, ...]]:
        if m == 1:
            yield (n,)
            return
        for head in range(1, n - m + 2):
            for tail in self._compositions(n - head, m - 1):
                yield (head,) + tail

    @staticmethod
    def _commutator(X: ArrayLike, Y: ArrayLike) -> ArrayLike:
        return X @ Y - Y @ X

    def _unpack(self, y: ArrayLike) -> ArrayLike:
        return y.reshape(self.K, self.dim, self.dim)

    def _pack(self, omega_terms: ArrayLike) -> ArrayLike:
        return omega_terms.reshape(-1)

    def rhs(self, t: float, y: ArrayLike) -> ArrayLike:
        omega = self._unpack(y)
        d_omega = np.zeros_like(omega)

        At = np.asarray(self.A(t), dtype=np.complex128)
        if At.shape != (self.dim, self.dim):
            raise ValueError(
                f"A(t) must return shape {(self.dim, self.dim)}, got {At.shape}"
            )

        d_omega[0] = At

        for n in range(2, self.K + 1):
            total = np.zeros((self.dim, self.dim), dtype=np.complex128)

            for j in range(1, n):
                Bj = self.bernoulli_coeffs[j]
                if Bj == 0:
                    continue
                coeff = Bj / factorial(j)

                for comp in self._compositions(n - 1, j):
                    nested = At
                    for idx in reversed(comp):
                        nested = self._commutator(omega[idx - 1], nested)
                    total = total + coeff * nested

            d_omega[n - 1] = total

        return self._pack(d_omega)


class HamiltonianMagnusSolver:
    """Compute first ``K`` Magnus terms for a time-dependent Hamiltonian ``H(t)``.

    Parameters
    ----------
    H:
        Callable returning a square complex matrix for each scalar time.
    dim:
        Hilbert-space dimension.
    include_minus_i:
        If True, uses A(t) = -1j * H(t). If False, treats H(t) directly as A(t).
    """

    def __init__(
        self,
        H: Callable[[float], ArrayLike],
        dim: int,
        include_minus_i: bool = True,
    ):
        self.H = H
        self.dim = int(dim)
        self.include_minus_i = bool(include_minus_i)

    def generator(self, t: float) -> ArrayLike:
        Ht = np.asarray(self.H(t), dtype=np.complex128)
        if Ht.shape != (self.dim, self.dim):
            raise ValueError(
                f"H(t) must return shape {(self.dim, self.dim)}, got {Ht.shape}"
            )
        return -1j * Ht if self.include_minus_i else Ht

    def solve(
        self,
        K: int,
        T: float,
        *,
        t0: float = 0.0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        method: str = "DOP853",
    ) -> MagnusResult:
        system = HamiltonianMagnusODE(self.generator, K=K, dim=self.dim)
        y0 = np.zeros((K, self.dim, self.dim), dtype=np.complex128).reshape(-1)

        sol = solve_ivp(
            system.rhs,
            t_span=(t0, T),
            y0=y0,
            t_eval=[T],
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Magnus ODE solve failed: {sol.message}")

        terms = sol.y[:, -1].reshape(K, self.dim, self.dim)
        return MagnusResult(terms=terms, t_final=T)
