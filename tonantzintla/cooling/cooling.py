# -*- coding: utf-8 -*-
"""Cooling source terms and cooling-curve helpers."""

from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import interp1d
from .. import eos
from .schure_tables import lambda_hd


def _get_S():
    try:
        from .. import settings as _S
        return _S
    except Exception:
        return None

try:
    from .. import settings as S
except Exception:
    S = None

try:
    from ..cinder import cinder_cool
except Exception:
    cinder_cool = None


def load_schure_table(path=None):
    S0 = S or _get_S()
    path = path or getattr(S0, "LAMBDA_TABLE_FILE", "Schure_total.dat")
    if not os.path.exists(path):
        return lambda T: np.zeros_like(T, dtype=float)
    arr = np.loadtxt(path)
    T_vals, Lam_vals = 10.0**arr[:, 0], 10.0**arr[:, 1]
    f = interp1d(np.log10(T_vals), np.log10(Lam_vals), kind="linear", fill_value="extrapolate", bounds_error=False)
    def Lambda(Tin):
        Tclip = np.clip(Tin, 1.0e2, 1.0e8)
        return 10.0**f(np.log10(Tclip))
    return Lambda


Lambda_T = load_schure_table()
mu_piecewise = eos.mu_from_temperature


def ne_over_nH_schure(T):
    _logT = np.array([
        3.80,3.84,3.88,3.92,3.96,4.00,4.04,4.08,4.12,4.16,4.20,4.24,4.28,4.32,4.36,4.40,4.44,4.48,4.52,4.56,4.60,4.64,4.68,4.72,4.76,4.80,4.84,4.88,4.92,4.96,5.00,5.04,5.08,5.12,5.16,5.20,5.24,5.28,5.32,5.36,5.40,5.44,5.48,5.52,5.56,5.60,5.64,5.68,5.72,5.76,5.80,5.84,5.88,5.92,5.96,
        6.00,6.04,6.08,6.12,6.16,6.20,6.24,6.28,6.32,6.36,6.40,6.44,6.48,6.52,6.56,6.60,6.64,6.68,6.72,6.76,6.80,6.84,6.88,6.92,6.96,7.00,7.04,7.08,7.12,7.16,7.20,7.24,7.28,7.32,7.36,7.40,7.44,7.48,7.52,7.56,7.60,7.64,7.68,7.72,7.76,7.80,7.84,7.88,7.92,7.96,8.00,8.04,8.08,8.12,8.16
    ])
    _ratio = np.array([
        1.3264e-05,4.2428e-05,8.8276e-05,1.7967e-04,8.4362e-04,3.4295e-03,1.3283e-02,4.2008e-02,1.2138e-01,3.0481e-01,5.3386e-01,7.6622e-01,8.9459e-01,9.5414e-01,9.8342e-01,1.0046,1.0291,1.0547,1.0767,1.0888,1.0945,1.0972,1.0988,1.1004,1.1034,1.1102,1.1233,1.1433,1.1638,1.1791,1.1885,1.1937,1.1966,1.1983,1.1993,1.1999,1.2004,1.2008,1.2012,1.2015,1.2020,1.2025,1.2030,1.2035,1.2037,1.2039,1.2040,1.2041,1.2042,1.2044,1.2045,1.2046,1.2047,1.2049,1.2050,
        1.2051,1.2053,1.2055,1.2056,1.2058,1.2060,1.2062,1.2065,1.2067,1.2070,1.2072,1.2075,1.2077,1.2078,1.2079,1.2080,1.2081,1.2082,1.2083,1.2083,1.2084,1.2084,1.2085,1.2085,1.2086,1.2086,1.2087,1.2087,1.2088,1.2088,1.2089,1.2089,1.2089,1.2089,1.2089,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2090,1.2091,1.2091,1.2091
    ])
    n = min(_logT.size, _ratio.size)
    return np.interp(np.log10(np.clip(T, 1.0e2, 1.0e8)), _logT[:n], _ratio[:n])


def apply_cooling(U, dt, bins, D_bins):
    dt = float(dt)
    if dt <= 0.0:
        return U
    sp = (slice(1, -1),) + tuple(slice(None) for _ in range(U[0].ndim - 1))
    Us = U[(slice(None),) + sp]
    rho, _, _, _, P = eos.prim_from_cons(Us)
    T = eos.temperature_from_pressure_density(P, rho)
    S0 = S or _get_S()
    XH = float(getattr(S0, "X_H", 0.71)) if S0 is not None else 0.71
    nH = XH * rho / eos.mH()
    q = ne_over_nH_schure(T) * nH * nH * Lambda_T(T)
    if bool(getattr(S0, "DUST_GAS_COOLING_ON", False)) and cinder_cool is not None and bins is not None and D_bins is not None:
        try:
            q = q + cinder_cool.dust_cooling_n2(rho/eos.mu_from_temperature(T), T, D_bins[sp + (slice(None),)], bins)
        except Exception:
            pass
    eint = np.maximum(eos.internal_energy(Us) - q * dt, 1.0e-30)
    Us[5] = eint if Us.shape[0] > 5 else Us[4]
    Us[4] = eos.kinetic_energy_density(Us) + eint
    U[(slice(None),) + sp] = Us
    return eos.reconcile_dual_energy(U)


def write_cooling_comparison_png(out_png, bins=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.logspace(2, 8, 400)
    L_gas = Lambda_T(T)
    if cinder_cool is not None:
        if bins is None:
            bins = cinder_cool.build_bins_from_settings()
        L_dust_per_D = cinder_cool.dust_lambda_per_D(T, bins=bins)
    else:
        L_dust_per_D = np.zeros_like(T)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(T, L_gas, label="Λ_gas (Schure)")
    ax.loglog(T, L_dust_per_D, label="Λ_dust per unit 𝓓")
    ax.set_xlabel("T [K]"); ax.set_ylabel("Λ [erg cm$^3$ s$^{-1}$]")
    ax.grid(True, which='both', ls=':')
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def cooling_dt_peek(U, bins=None, D_bins=None):
    sp = (slice(1, -1),) + tuple(slice(None) for _ in range(U[0].ndim - 1))
    Us = U[(slice(None),) + sp]
    rho, _, _, _, P = eos.prim_from_cons(Us)
    T = eos.temperature_from_pressure_density(P, rho)
    S0 = S or _get_S()
    XH = float(getattr(S0, "X_H", 0.71)) if S0 is not None else 0.71
    nH = XH * rho / eos.mH()
    q = np.maximum(ne_over_nH_schure(T) * nH * nH * Lambda_T(T), 1.0e-99)
    return float(np.nanmin(np.maximum(eos.internal_energy(Us) / q, 1.0)))
