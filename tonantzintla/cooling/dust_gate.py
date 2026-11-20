import numpy as np

_ENVELOPE = None  # Cache: {'logT_grid','L_env','D_ref','Tmin','Tmax'}

def _build_envelope(cinder_cool_func, S, logT_min=4.0, logT_max=8.5, npts=48):
    """Build an upper envelope Lambda_dust(logT) for the configured dust distribution.
    Assumes Lambda_dust scales ~linearly with total dust mass (optically thin).
    Returns dict with logT_grid and L_env (erg s^-1 cm^3) per unit D_ref.
    """
    logT = np.linspace(logT_min, logT_max, npts, dtype=float)
    T = 10.0**logT
    rho_ref = 1.0  # irrelevant for Lambda units (erg s^-1 cm^3)

    try:
        D_ref = float(getattr(S, 'D_TOTAL_INIT', 1.0))
    except Exception:
        D_ref = 1.0

    L = np.zeros_like(T, dtype=float)
    for i,Ti in enumerate(T):
        try:
            L[i] = float(cinder_cool_func(Ti, rho_ref, S))
        except Exception:
            L[i] = 0.0

    L_env = L / D_ref if D_ref > 0.0 else L

    nz = np.where(L_env > 0)[0]
    if nz.size > 0:
        Tmin = logT[nz[0]]; Tmax = logT[nz[-1]]
    else:
        Tmin = logT_min; Tmax = logT_min

    return dict(logT_grid=logT, L_env=L_env, D_ref=D_ref, Tmin=Tmin, Tmax=Tmax)

def ensure_envelope(cinder_cool_func, S):
    global _ENVELOPE
    if _ENVELOPE is None:
        _ENVELOPE = _build_envelope(cinder_cool_func, S)
    return _ENVELOPE

def dust_skip_mask(T, D_local, L_gas, S, cinder_cool_func):
    """Return boolean mask where dust cooling can be skipped safely."""
    env = ensure_envelope(cinder_cool_func, S)
    logT = np.log10(np.asarray(T, dtype=float))
    L_g = np.asarray(L_gas, dtype=float)
    D   = np.asarray(D_local, dtype=float)

    frac = float(getattr(S, 'DUST_COOL_SKIP_FRAC', 1e-3))
    Dmin = float(getattr(S, 'DUST_MIN_RATIO', 1e-8))

    skip = (D <= Dmin) | (logT < env['Tmin']) | (logT > env['Tmax'])

    if np.any(~skip):
        Le = np.interp(logT[~skip], env['logT_grid'], env['L_env'], left=0.0, right=0.0)
        Le_scaled = Le * D[~skip]
        skip_part = Le_scaled < (frac * L_g[~skip])
        tmp = skip.copy()
        tmp[~skip] |= skip_part
        skip = tmp

    return skip
