# -*- coding: utf-8 -*-
"""
FLASH-like Schure(2009)+DMC72 cooling tables.
Return Λ_hd(T) = (n_e/n_H)*Λ_N(T), so use q_gas = n_H**2 * Λ_hd(T).
"""
from __future__ import annotations
import numpy as np
import os

_LOGT_LOW = None     # (NL,)
_LAM_LOW = None      # (NL,4)   log10 Λ_hd(T) for fi=[1e-4,1e-3,1e-2,1e-1]
_LOGT_HIGH = None    # (NH,)
_LAM_ELEM = None     # (NH,15)  log10 Λ_N per species
_NE_OVER_NH = None   # (NH,)    linear

def _isnum(s):
    try:
        float(s.replace('D','E'))
        return True
    except Exception:
        return False

def _load_low(path):
    global _LOGT_LOW, _LAM_LOW
    lines = open(path,'r',errors='ignore').read().splitlines()
    Ts=[]; cols=[]
    for ln in lines:
        s=ln.strip()
        if not s or s[0] in '#!;': continue
        xs = s.replace(',', ' ').split()
        nums=[float(t.replace('D','E')) for t in xs if _isnum(t)]
        if len(nums)>=5:
            Ts.append(nums[0]); cols.append(nums[1:5])
    _LOGT_LOW = np.asarray(Ts,float)
    _LAM_LOW  = np.asarray(cols,float)
    if _LAM_LOW.shape[1]!=4: raise ValueError("DMC72 table parse error")

def _load_high(path):
    global _LOGT_HIGH, _LAM_ELEM, _NE_OVER_NH
    lines = open(path,'r',errors='ignore').read().splitlines()
    Ts=[]; elem=[]; ratio=[]
    for ln in lines:
        s=ln.strip()
        if not s or s[0] in '#!;': continue
        xs=s.replace(',', ' ').split()
        nums=[float(t.replace('D','E')) for t in xs if _isnum(t)]
        if len(nums)>=17:
            Ts.append(nums[0])
            elem.append(nums[1:16])   # 15 species
            ratio.append(nums[16])
    _LOGT_HIGH  = np.asarray(Ts,float)
    _LAM_ELEM   = np.asarray(elem,float)
    _NE_OVER_NH = np.asarray(ratio,float)
    if _LAM_ELEM.shape[1]!=15: raise ValueError("Schure09 table parse error")

def _ensure_loaded(path_high, path_low):
    if _LOGT_HIGH is None or _LOGT_LOW is None:
        if not (os.path.exists(path_high) and os.path.exists(path_low)):
            raise FileNotFoundError("Schure tables not found")
        _load_low(path_low); _load_high(path_high)

def _interp_low(logT):
    # returns bilinear neighbors for columns (NL,4)
    x = (np.clip(logT, _LOGT_LOW[0], _LOGT_LOW[-1]) - _LOGT_LOW[0]) / max(_LOGT_LOW[1]-_LOGT_LOW[0], 1e-12)
    i0 = np.clip(np.floor(x).astype(int), 0, len(_LOGT_LOW)-2)
    fx = np.clip(x - i0, 0.0, 1.0)
    lam_i   = _LAM_LOW[i0,   :]
    lam_ip1 = _LAM_LOW[i0+1, :]
    return lam_i, lam_ip1, fx

def _iondeg_index_from_nH(nH):
    # approximate FLASH co_ionDeg-based mapping
    log_nH = np.log10(np.maximum(nH, 1e-99))
    log_xe = np.where(log_nH < -3.0, -1.0,
              np.where(log_nH < -1.5, -1.0 - 0.5*(log_nH + 3.0),
              np.where(log_nH < 0.5,  -1.75 - 1.125*(log_nH + 1.5), -4.0)))
    j = np.floor(log_xe + 4.0).astype(int) + 1     # 1..4
    j = np.clip(j, 1, 3)                            # use j and j+1
    p = (log_xe + 4.0) - np.floor(log_xe + 4.0)
    return j, p

def lambda_hd(T, nH, path_high, path_low, ionDegInd=0, rel_weights=None):
    """
    Compute Λ_hd(T) using DMC72 (low T) + Schure09 (high T).
    rel_weights: length-15 scaling vector for per-element Λ_N (He + metals),
                 defaults to ones (solar). Only used in high-T Schure region.
    """
    _ensure_loaded(path_high, path_low)
    T = np.asarray(T,float); nH = np.asarray(nH,float)
    logT = np.log10(np.maximum(T,1.0))
    out = np.zeros_like(logT)

    lTmin_low  = _LOGT_LOW[0]
    lTmax_low  = _LOGT_LOW[-1]
    lTmin_high = _LOGT_HIGH[0]

    # 1) low-T DMC72
    mlow = logT <= lTmax_low
    if np.any(mlow):
        lam_i, lam_ip1, fx = _interp_low(logT[mlow])
        if ionDegInd and (1 <= ionDegInd <= 4):
            col = ionDegInd - 1
            val = (1.0 - fx)*lam_i[:,col] + fx*lam_ip1[:,col]
            out[mlow] = (10.0**val) * (float(rel_weights[1]) if rel_weights is not None else 1.0)
        else:
            j, p = _iondeg_index_from_nH(nH[mlow])
            j0 = j - 1
            val = ((1.0 - fx) * (1.0 - p) * lam_i[np.arange(len(j0)), j0] +
                   (fx)       * (1.0 - p) * lam_ip1[np.arange(len(j0)), j0] +
                   (1.0 - fx) * (p)       * lam_i[np.arange(len(j0)), j0+1] +
                   (fx)       * (p)       * lam_ip1[np.arange(len(j0)), j0+1])
            out[mlow] = 10.0**val

    # 2) blend region
    mmid = (logT > lTmax_low) & (logT < lTmin_high)
    if np.any(mmid):
        lam_low_max = _LAM_LOW[-1,3]   # fi=1e-1
        rel = np.ones(15,float) if rel_weights is None else np.asarray(rel_weights,float)
        Lsch0 = np.sum(10.0**_LAM_ELEM[0,:] * rel)
        q = (logT[mmid] - lTmax_low) / max(lTmin_high - lTmax_low, 1e-12)
        out[mmid] = (1.0 - q) * (10.0**lam_low_max) + q * Lsch0

    # 3) high-T Schure
    mhi = logT >= lTmin_high
    if np.any(mhi):
        rel = np.ones(15,float) if rel_weights is None else np.asarray(rel_weights,float)
        v = (np.clip(logT[mhi], _LOGT_HIGH[0], _LOGT_HIGH[-1]) - _LOGT_HIGH[0]) / max(_LOGT_HIGH[1]-_LOGT_HIGH[0], 1e-12)
        i0 = np.clip(np.floor(v).astype(int), 0, len(_LOGT_HIGH)-2)
        fv = np.clip(v - i0, 0.0, 1.0)
        Llo = np.sum(10.0**_LAM_ELEM[i0,   :] * rel, axis=1)
        Lhi = np.sum(10.0**_LAM_ELEM[i0+1, :] * rel, axis=1)
        rlo = _NE_OVER_NH[i0]; rhi = _NE_OVER_NH[i0+1]
        out[mhi] = (1.0 - fv) * Llo * rlo + fv * Lhi * rhi
    return out