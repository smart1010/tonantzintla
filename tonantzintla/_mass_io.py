# Mass diagnostics and snapshot I/O helpers
import os
import numpy as np
from . import eos

__all__ = [
    "ensure_mass_header",
    "compute_phase_masses_local",
    "reduce6",
    "append_mass_row_once",
    "snap_exists",
    "last_time_kyr",
    "should_write_mass",
]

def _header_line():
    fields = ["t_kyr","Mg","Md","Gc","Gw","Gh","Dc","Dw","Dh"]
    return "# " + f"{fields[0]:>10}" + "".join(f" {name:>14}" for name in fields[1:]) + "\n"
def ensure_mass_header(path: str) -> None:
    """Ensure a nine-column header while preserving data rows."""
    header = _header_line()
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w") as fh:
            fh.write(header)
        return
    try:
        with open(path, "r") as fh:
            lines = fh.readlines()
        # Find first non-empty line
        first = ""
        idx = 0
        while idx < len(lines):
            if lines[idx].strip():
                first = lines[idx]
                break
            idx += 1
        if first.lstrip().startswith("#"):
            if first.strip() == header.strip():
                return
            # Replace header comments while preserving data rows.
            j = 0
            while j < len(lines) and lines[j].lstrip().startswith("#"):
                j += 1
            with open(path, "w") as fh:
                fh.write(header)
                fh.writelines(lines[j:])
            return
        else:
            # No header: insert once at top
            with open(path, "w") as fh:
                fh.write(header)
                fh.writelines(lines)
    except Exception:
        pass

def _sum_bins_to_ratio(D_sub: np.ndarray, rho_shape) -> np.ndarray:
    """Sum dust-to-gas ratios across bins -> one scalar per cell (dimensionless)."""
    D = np.clip(D_sub, 0, None)
    # Try common layouts
    if D.ndim == 4:
        if D.shape[0] not in rho_shape:     # (bins,x,y,z)
            cand = D.sum(axis=0)
            if tuple(cand.shape) == tuple(rho_shape):
                return cand
        if D.shape[-1] not in rho_shape:    # (x,y,z,bins)
            cand = D.sum(axis=-1)
            if tuple(cand.shape) == tuple(rho_shape):
                return cand
    # fallback: search an axis whose sum matches rho shape
    for ax in range(D.ndim):
        cand = D.sum(axis=ax)
        if tuple(cand.shape) == tuple(rho_shape):
            return cand
    cand = D.sum(axis=0)
    rx, ry, rz = rho_shape
    cx, cy, cz = cand.shape
    return cand[:min(rx,cx), :min(ry,cy), :min(rz,cz)]

def compute_phase_masses_local(U_sub, D_bins_sub, dx, dy, dz, GAMMA, kB, mH, mu_mol, T_bounds):
    """Return (gas_c, gas_w, gas_h, dust_c, dust_w, dust_h) in grams for this subdomain."""
    rho, vx, vy, vz, P = eos.prim_from_cons(U_sub)

    T1, T2 = T_bounds
    T = eos.temperature_from_pressure_density(P, rho)

    cold = (T < T1)
    warm = (T >= T1) & (T < T2)
    hot  = (T >= T2)

    vol = float(dx) * float(dy) * float(dz)
    rho = np.asarray(rho, dtype=np.float64)
    m_g_cell = rho * vol
    g_c = float(m_g_cell[cold].sum())
    g_w = float(m_g_cell[warm].sum())
    g_h = float(m_g_cell[hot].sum())

    if D_bins_sub is None:
        return g_c, g_w, g_h, 0.0, 0.0, 0.0

    d_ratio_cell = np.asarray(_sum_bins_to_ratio(D_bins_sub, rho.shape), dtype=np.float64)
    m_d_cell     = d_ratio_cell * m_g_cell
    d_c = float(m_d_cell[cold].sum())
    d_w = float(m_d_cell[warm].sum())
    d_h = float(m_d_cell[hot].sum())
    return g_c, g_w, g_h, d_c, d_w, d_h

def reduce6(COMM, vals):
    g_c, g_w, g_h, d_c, d_w, d_h = vals
    Gc = COMM.reduce(g_c, root=0)
    Gw = COMM.reduce(g_w, root=0)
    Gh = COMM.reduce(g_h, root=0)
    Dc = COMM.reduce(d_c, root=0)
    Dw = COMM.reduce(d_w, root=0)
    Dh = COMM.reduce(d_h, root=0)
    return Gc, Gw, Gh, Dc, Dw, Dh

def _last_time_kyr(path: str):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return None
            back = min(65536, size)
            f.seek(-back, os.SEEK_END)
            tail = f.read().decode("utf-8", "ignore").splitlines()
        for line in reversed(tail):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()[0]
            try:
                return float(tok)
            except Exception:
                continue
    except Exception:
        return None

def last_time_kyr(path: str):
    return _last_time_kyr(path)

def should_write_mass(MASS_FILE: str, t: float, YR_S: float) -> bool:
    """Compatibility helper: True if this time step hasn't been recorded yet."""
    t_kyr = t / YR_S / 1e3
    last = _last_time_kyr(MASS_FILE)
    return (last is None) or (t_kyr > last + 1e-15)

def append_mass_row_once(path, t_kyr, Mg_Msun, Md_Msun, Gc_g, Gw_g, Gh_g, Dc_g, Dw_g, Dh_g, MSUN):
    """Append one formatted row; enforce strictly increasing time; aligned columns."""
    ensure_mass_header(path)
    last = _last_time_kyr(path)
    if (last is not None) and (t_kyr <= last + 1e-15):
        return
    fmt = "{:12.5f}" + (" {:>14.8e}" * 8) + "\n"
    with open(path, "a") as f:
        f.write(fmt.format(
            t_kyr, Mg_Msun, Md_Msun,
            Gc_g/MSUN, Gw_g/MSUN, Gh_g/MSUN,
            Dc_g/MSUN, Dw_g/MSUN, Dh_g/MSUN
        ))

def snap_exists(outdir, basen, snap_id: int) -> bool:
    stem = f"{snap_id:04d}"
    paths = [
        os.path.join(outdir, f"snap_{basen}_{stem}.png"),
        os.path.join(outdir, f"asc-xy-{basen}_{stem}"),
        os.path.join(outdir, f"asc-xz-{basen}_{stem}"),
        os.path.join(outdir, f"asc-yz-{basen}_{stem}"),
        os.path.join(outdir, f"3D_{basen}_hdf5_{stem}"),
        os.path.join(outdir, f"3D_{basen}_hdf5_chk_{stem}"),
        os.path.join(outdir, f"{basen}_{stem}"),
    ]
    def _ok(p):
        if not os.path.exists(p):
            return False
        base = os.path.basename(p)
        if "hdf5" in base:
            try:
                return os.path.getsize(p) > 0
            except Exception:
                return False
        return True
    return any(_ok(p) for p in paths)
