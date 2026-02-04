import numpy as np
from scipy.integrate import solve_ivp, DOP853, OdeSolution
from scipy.linalg import expm, logm
from typing import Callable, Optional

def ancilla_projection(Omega: np.ndarray, sigma: np.ndarray):
    d = Omega.shape[0] // 2
    Omega_rs = Omega.reshape(d, 2, d, 2)
    out = np.zeros((d, d), dtype=Omega.dtype)
    for a in range(2):
        for b in range(2): out += Omega_rs[:, a, :, b] * sigma[b, a]
    return out / 2

def inner_product(X : np.ndarray, Y : np.ndarray): return np.trace(X.conj().T @ Y)

def test_similarity(H: np.ndarray, A: np.ndarray, coef: float):
    c = inner_product(H, A) / inner_product(H, H)
    D = A - c * H

    dir_error = np.linalg.norm(D, 'fro') / np.linalg.norm(A, 'fro')

    if abs(coef) < 1e-14: coef_error = abs(c)
    else: coef_error = abs(c - coef) / abs(coef)


    return dir_error, coef_error



def test_curve( H : np.ndarray  , omega : Callable[[float], float] , interval : np.ndarray , expected_poly : np.ndarray , lambdas = [0.05, 0.1, 0.15, 0.2, 0.25] ):
    assert len(H.shape) == 2 and H.shape[0] == H.shape[1]
    assert len(interval.shape) == 1
    assert len(expected_poly.shape) == 2 and expected_poly.shape[0] == 3

    X = np.asarray([ [0, 1], [1, 0] ])
    Y = np.asarray([ [0, -1j], [1j, 0] ])
    Z = np.asarray([ [1, 0], [0, -1] ])
    t0 = interval[0]
    tf = interval[-1]

    omega_integral : OdeSolution = solve_ivp( lambda t, y: omega(t), (t0, tf), [0.0], 'DOP853', dense_output=True ).sol

    samples = {'x': [], 'y': [], 'z': []}

    u_prime_shape = 2 * H.shape[0]

    for lamb in lambdas:

        updated_H = lamb * H

        def ode(time: float, unitary_flat: np.ndarray):
            U = unitary_flat.reshape(u_prime_shape, u_prime_shape)
            omega_integral_value = omega_integral(time)
            curve_component = np.cos(2 * omega_integral_value) * Z + np.sin(2 * omega_integral_value) * Y
            dU = -1j * np.kron(updated_H, curve_component) @ U
            return dU.reshape(-1)

        U_T = solve_ivp( ode, (t0, tf), np.eye(u_prime_shape).reshape(-1), 'DOP853', t_eval=[tf] ).y[:, 0].reshape(u_prime_shape, u_prime_shape)
        
        unitary_omega = logm(U_T)

        samples['x'].append(ancilla_projection(unitary_omega, X))
        samples['y'].append(ancilla_projection(unitary_omega, Y))
        samples['z'].append(ancilla_projection(unitary_omega, Z))

    degree = expected_poly.shape[1]

    design_X = np.asarray( [ [ lam**(k+1) for k in range(degree) ] for lam in lambdas ] )

    data = np.asarray( [[sample.reshape(-1) for sample in samples[pauli]] for pauli in ['x', 'y', 'z']] )  

    d = H.shape[0]

    operators : np.ndarray = np.linalg.lstsq(design_X, data).x.reshape(3, degree, d, d)

    errors = np.asarray([ 
        [
            test_similarity( np.linalg.matrix_power(H, k + 1), A, expected_poly[pauli, k]) 
            for k, A in enumerate(pauli_operators)
        ]
        for pauli, pauli_operators in enumerate(operators)
    ])

    return operators, errors


def test_fidelity( H : np.ndarray, omega : Callable[[float], float], interval : np.ndarray, expected_unitary: np.ndarray ):
    assert len(H.shape) == 2 and H.shape[0] == H.shape[1]
    dU_shape = 2 * H.shape[0]
    assert len(expected_unitary.shape) == 2 and expected_unitary.shape == (dU_shape, dU_shape)

    t0, tf = interval 
    X = np.asarray([ [0, 1], [1, 0] ])
    Y = np.asarray([ [0, -1j], [1j, 0] ])
    Z = np.asarray([ [1, 0], [0, -1] ])

    omega_integral : OdeSolution = solve_ivp( lambda t, y: omega(t), (t0, tf), [0.0], 'DOP853', dense_output=True ).sol


    def ode(time: float, unitary_flat: np.ndarray):
        unitary = unitary_flat.reshape((dU_shape, dU_shape))
        omega_integral_value = omega_integral(time)[0]
        curve_component = np.cos(2 * omega_integral_value) * Z + np.sin(2 * omega_integral_value) * Y
        dU = -1j * np.kron(H, curve_component) @ unitary
        return dU.reshape(-1)

    U_T = solve_ivp( ode, (t0, tf), np.eye( dU_shape, dtype=complex ).reshape(-1), 'DOP853', t_eval=[tf] ).y[..., 0].reshape((dU_shape, dU_shape))

    fidelity = np.abs(np.trace( U_T @ expected_unitary.conj().T )**2) / dU_shape**2
    
    return fidelity