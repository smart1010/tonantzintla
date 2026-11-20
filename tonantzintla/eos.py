import numpy as np


def _settings():
    try:
        from . import settings as S
        return S
    except Exception:
        return None


def _const(name, default):
    S = _settings()
    return float(getattr(S, name, default)) if S is not None else float(default)


def gamma():
    return _const("gamma", 5.0/3.0)


def kB():
    return _const("kB", 1.3806485e-16)


def mH():
    return _const("mH", 1.6726219e-24)


def mu_molecular():
    return 2.33 * mH()


def mu_atomic():
    return 1.27 * mH()


def mu_ionized():
    return 0.61 * mH()


def mu_from_temperature(T):
    T = np.asarray(T, dtype=float)
    mu_mol = mu_molecular()
    mu_atm = mu_atomic()
    mu_ion = mu_ionized()
    mu = np.empty_like(T, dtype=float)
    mu[T <= 300.0] = mu_mol
    m = (T > 300.0) & (T < 700.0)
    mu[m] = mu_mol + (mu_atm - mu_mol) * ((T[m] - 300.0) / 400.0)
    m = (T >= 700.0) & (T <= 9300.0)
    mu[m] = mu_atm
    m = (T > 9300.0) & (T < 9700.0)
    mu[m] = mu_atm + (mu_ion - mu_atm) * ((T[m] - 9300.0) / 400.0)
    mu[T >= 9700.0] = mu_ion
    return float(mu) if mu.ndim == 0 else mu


def temperature_from_pressure_density(P, rho, niter=5, T_floor=10.0):
    rho = np.maximum(np.asarray(rho, dtype=float), 1.0e-30)
    P = np.maximum(np.asarray(P, dtype=float), 0.0)
    T = np.maximum(P * mu_ionized() / (rho * kB()), T_floor)
    for _ in range(int(niter)):
        T = np.maximum(P * mu_from_temperature(T) / (rho * kB()), T_floor)
    return T


def pressure_from_density_temperature(rho, T, niter=5):
    T = np.maximum(np.asarray(T, dtype=float), 10.0)
    return np.maximum(np.asarray(rho, dtype=float), 1.0e-30) * kB() * T / mu_from_temperature(T)


def kinetic_energy_density(U):
    rho = np.maximum(U[0], 1.0e-30)
    vx = U[1] / rho
    vy = U[2] / rho
    vz = U[3] / rho
    return 0.5 * rho * (vx*vx + vy*vy + vz*vz)


def internal_energy_from_total(U):
    return np.maximum(U[4] - kinetic_energy_density(U), 1.0e-30)


def internal_energy(U, eta=1.0e-3):
    eint_total = internal_energy_from_total(U)
    if U.shape[0] <= 5:
        return eint_total
    Etot = np.maximum(np.abs(U[4]), 1.0e-30)
    return np.where(eint_total / Etot > eta, eint_total, np.maximum(U[5], 1.0e-30))


def pressure_from_cons(U, eta=1.0e-3):
    return (gamma() - 1.0) * internal_energy(U, eta=eta)


def prim_from_cons(U, eta=1.0e-3):
    rho = np.maximum(U[0], 1.0e-30)
    return rho, U[1]/rho, U[2]/rho, U[3]/rho, pressure_from_cons(U, eta=eta)


def append_internal_energy(U):
    if U.shape[0] > 5:
        return U
    return np.concatenate([U, internal_energy_from_total(U)[None, ...].astype(U.dtype, copy=False)], axis=0)


def reconcile_dual_energy(U, eta=1.0e-3):
    if U.shape[0] <= 5:
        return U
    ek = kinetic_energy_density(U)
    eint_total = np.maximum(U[4] - ek, 1.0e-30)
    good = (eint_total / np.maximum(np.abs(U[4]), 1.0e-30)) > eta
    U[5] = np.where(good, eint_total, np.maximum(U[5], 1.0e-30))
    U[4] = np.where(good, U[4], ek + U[5])
    return U


def temperature_from_cons(U, eta=1.0e-3):
    return temperature_from_pressure_density(pressure_from_cons(U, eta=eta), U[0])
