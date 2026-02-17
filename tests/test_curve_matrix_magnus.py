import unittest


try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from base_curve import LearnableCurve
    from magnus import compute_R_terminal
    from magnus_hamiltonian import HamiltonianMagnusSolver
except Exception:  # pragma: no cover
    LearnableCurve = None
    compute_R_terminal = None
    HamiltonianMagnusSolver = None


DEPS_OK = all(x is not None for x in (np, torch, LearnableCurve, compute_R_terminal, HamiltonianMagnusSolver))


if DEPS_OK:
    class TestCurveMagnusMatchesMatrixMagnus(unittest.TestCase):
        class AnalyticDriveCurve(LearnableCurve):
            """Curve-like object where curve_d1(t) is the drive r(t)."""

            def __init__(self, drive_fn, t0=0.0, tf=1.0):
                super().__init__(interval=[t0, tf])
                self.drive_fn = drive_fn

            def curve(self, t):
                return self.curve_d1(t)

            def curve_d1(self, t):
                t = torch.as_tensor(t, dtype=torch.float32)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                values = [self.drive_fn(float(tt.item())) for tt in t]
                return torch.stack(values, dim=0)

            def curve_d2(self, t):
                t = torch.as_tensor(t, dtype=torch.float32)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                dt = 1e-5
                vals_p = [self.drive_fn(float(tt.item() + dt)) for tt in t]
                vals_m = [self.drive_fn(float(tt.item() - dt)) for tt in t]
                return (torch.stack(vals_p, dim=0) - torch.stack(vals_m, dim=0)) / (2 * dt)

        @staticmethod
        def _paulis():
            sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            return sx, sy, sz

        @classmethod
        def _hamiltonian_from_drive(cls, drive_fn):
            sx, sy, sz = cls._paulis()

            def H(t: float):
                r = drive_fn(float(t)).detach().cpu().numpy().astype(np.complex128)
                return r[0] * sx + r[1] * sy + r[2] * sz

            return H

        @classmethod
        def _matrix_terms_to_vectors(cls, terms):
            sx, sy, sz = cls._paulis()
            basis = [sx, sy, sz]
            out = np.zeros((terms.shape[0], 3), dtype=np.complex128)
            for n in range(terms.shape[0]):
                for i, sigma in enumerate(basis):
                    out[n, i] = 0.5 * np.trace(sigma @ terms[n])
            return out

        def _assert_curve_vs_matrix(self, drive_fn, K=4, T=1.0, tol=3e-4):
            curve = self.AnalyticDriveCurve(drive_fn, t0=0.0, tf=T)
            curve_terms = compute_R_terminal(curve, K=K, T=T, steps=2).detach().cpu().numpy()

            solver = HamiltonianMagnusSolver(self._hamiltonian_from_drive(drive_fn), dim=2, include_minus_i=False)
            matrix_terms = solver.solve(K=K, T=T).terms
            vector_terms = self._matrix_terms_to_vectors(matrix_terms)

            max_err = np.max(np.abs(curve_terms - vector_terms))
            self.assertLessEqual(max_err, tol, msg=f"max component error {max_err} exceeded tolerance {tol}")

        def test_constant_drive(self):
            vec = torch.tensor([0.35, -0.5, 0.2], dtype=torch.float32)

            def drive(_t: float):
                return vec

            self._assert_curve_vs_matrix(drive_fn=drive, K=5, T=0.9, tol=1e-6)

        def test_smooth_time_dependent_drive(self):
            def drive(t: float):
                return torch.tensor(
                    [
                        0.4 * np.cos(1.2 * t),
                        -0.3 * np.sin(0.8 * t + 0.2),
                        0.2 + 0.1 * np.cos(0.5 * t),
                    ],
                    dtype=torch.float32,
                )

            self._assert_curve_vs_matrix(drive_fn=drive, K=4, T=1.0, tol=2e-3)
else:
    class TestCurveMagnusMatchesMatrixMagnus(unittest.TestCase):
        @unittest.skip("requires numpy, scipy, torch, and torchdiffeq in this environment")
        def test_missing_dependencies(self):
            pass


if __name__ == "__main__":
    unittest.main()
