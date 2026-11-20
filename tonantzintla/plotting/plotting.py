import json
import logging
from ..hydro.solvers import prim_from_cons
from .. import eos
try:
    from ..settings import XMIN_PC, XMAX_PC, YMIN_PC, YMAX_PC, SNAP_EVERY, YR_S
except Exception:
    pass
import os
try:
    from ..settings import *
except Exception:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import h5py

import tempfile

def _atomic_write(target_path, writer_fn):
    d = os.path.dirname(target_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    ok = False
    try:
        writer_fn(tmp)
        os.replace(tmp, target_path)
        ok = True
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
    return ok

def _h5_stream_write(h, name, A, chunks_last=64):
    import numpy as _np
    A = _np.asarray(A)
    if A.ndim == 0:
        h.create_dataset(name, data=A)
        return
    shape = A.shape
    if A.ndim == 3:
        chunks = (shape[0], shape[1], min(shape[2], chunks_last))
    elif A.ndim == 2:
        chunks = (shape[0], min(shape[1], chunks_last))
    elif A.ndim == 1:
        chunks = (min(shape[0], chunks_last),)
    else:
        h.create_dataset(name, data=A)
        return
    dset = h.create_dataset(name, shape=shape, dtype=A.dtype, chunks=chunks)
    B = chunks[-1] if isinstance(chunks, tuple) else chunks[0]
    if A.ndim == 1:
        for i0 in range(0, shape[0], B):
            i1 = min(i0+B, shape[0])
            dset[i0:i1] = A[i0:i1]
    elif A.ndim == 2:
        for j0 in range(0, shape[1], B):
            j1 = min(j0+B, shape[1])
            dset[:, j0:j1] = A[:, j0:j1]
    elif A.ndim == 3:
        for k0 in range(0, shape[2], B):
            k1 = min(k0+B, shape[2])
            dset[:, :, k0:k1] = A[:, :, k0:k1]

# ---------------- compact colorbar ----------------
def _inset_colorbar(fig, ax, im, label):
    bbox = ax.get_position()
    width = CB_WIDTH_REL * bbox.width
    pad   = CB_PAD_REL   * bbox.width
    cax = fig.add_axes(Bbox.from_extents(bbox.x1 + pad, bbox.y0,
                                         bbox.x1 + pad + width, bbox.y1))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    return cb

# --------------- helpers -----------------
def _extent_1d(a0, a1, n):
    return np.linspace(a0, a1, n, endpoint=False) + 0.5*(a1-a0)/n

def _frame_offset_pc(axis):
    if str(globals().get("FRAME_AXIS_RUNTIME", globals().get("MOVING_FRAME_AXIS", "z"))).lower() == axis:
        return float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)) / PC_CM
    return 0.0

def _axis_bounds(axis):
    if axis == "x":
        a0, a1 = XMIN_PC, XMAX_PC
    elif axis == "y":
        a0, a1 = YMIN_PC, YMAX_PC
    else:
        a0, a1 = ZMIN_PC, ZMAX_PC
    off = _frame_offset_pc(axis)
    return a0 + off, a1 + off

def _field_limits(A, log=False, percentiles=(0.5, 99.5), symmetric=False,
                  min_span=0.0, padding=0.05, pad_abs=0.0,
                  lower_bound=None, upper_bound=None):
    B = np.asarray(A, dtype=float)
    if log:
        B = B[np.isfinite(B) & (B > 0.0)]
        B = np.log10(B)
    else:
        B = B[np.isfinite(B)]
    if B.size == 0:
        return [-40.0, -39.0] if log else [0.0, 1.0]
    if symmetric:
        vmax = float(np.percentile(np.abs(B), percentiles[1]))
        vmax = max(vmax * (1.0 + padding), 1.0e-12)
        return [-vmax, vmax]
    v0, v1 = (float(x) for x in np.percentile(B, percentiles))
    span = v1 - v0
    if span < min_span:
        centre = 0.5 * (v0 + v1)
        v0 = centre - 0.5 * min_span
        v1 = centre + 0.5 * min_span
    else:
        pad = max(padding * span, pad_abs)
        v0 -= pad
        v1 += pad
    if lower_bound is not None:
        v0 = max(v0, float(lower_bound))
    if upper_bound is not None:
        v1 = min(v1, float(upper_bound))
    if v0 >= v1:
        delta = max(min_span, abs(v0) * 1.0e-6, 1.0e-6)
        v0, v1 = v0 - 0.5 * delta, v1 + 0.5 * delta
    return [v0, v1]


_LOG_LIMIT_KEYS = frozenset((
    "logrho", "logT", "logeint", "logekin", "logetherm", "logD",
))


def _round_limits_outward(values, step=0.5):
    lo, hi = (float(values[0]), float(values[1]))
    scale = float(step)
    lo = scale * np.floor(lo / scale)
    hi = scale * np.ceil(hi / scale)
    if hi <= lo:
        hi = lo + scale
    return [float(lo), float(hi)]


def _plot_limits(F):
    limits = {
        "logrho": _field_limits(F["rho"], log=True, min_span=1.0, pad_abs=0.3),
        "logT": _field_limits(F["T"], log=True, min_span=1.0, pad_abs=0.5),
        "vmag": _field_limits(
            np.sqrt(F["vx"]**2 + F["vy"]**2 + F["vz"]**2) / 1.0e5,
            percentiles=(0.0, 99.5), min_span=1.0, lower_bound=0.0,
        ),
        "logeint": _field_limits(F["eint"], log=True, min_span=2.0, pad_abs=0.5),
        "logekin": _field_limits(F["ekin"], log=True, min_span=2.0, pad_abs=0.5),
        "logetherm": _field_limits(
            F["etherm_frac"], log=True, min_span=1.0, pad_abs=0.1, upper_bound=0.0,
        ),
        "vorticity": _field_limits(
            F["vorticity_myr"], percentiles=(0.0, 99.5), symmetric=True,
        ),
    }
    if F["D"] is not None:
        limits["logD"] = _field_limits(
            np.clip(F["D"], 1.0e-49, None), log=True, min_span=1.0, pad_abs=0.5,
        )
    for key in _LOG_LIMIT_KEYS.intersection(limits):
        limits[key] = _round_limits_outward(limits[key], step=0.5)
    return limits


_PLOT_LIMITS_CACHE = None
_PLOT_LIMITS_CACHE_PATH = None


def _plot_limits_path():
    return os.path.join(OUTDIR, f"plot_limits_{BASEN}.json")


def _load_plot_limits():
    path = _plot_limits_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError, TypeError):
        logging.warning(f"[plot] unable to read fixed color limits: {path}")
        return None
    limits = {}
    for key, values in raw.items():
        if isinstance(values, list) and len(values) == 2:
            pair = [float(values[0]), float(values[1])]
            if np.all(np.isfinite(pair)) and pair[0] < pair[1]:
                limits[key] = pair
    return limits or None


def _save_plot_limits(limits):
    path = _plot_limits_path()
    def _write(tmp_path):
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(limits, handle, indent=2, sort_keys=True)
            handle.write("\n")
    _atomic_write(path, _write)


def _fixed_plot_limits(F=None, initialize=False):
    global _PLOT_LIMITS_CACHE, _PLOT_LIMITS_CACHE_PATH
    path = _plot_limits_path()
    if _PLOT_LIMITS_CACHE_PATH != path:
        _PLOT_LIMITS_CACHE = None
        _PLOT_LIMITS_CACHE_PATH = path
    if initialize:
        if F is None:
            raise ValueError("Initial plotting fields are required to define fixed color limits")
        _PLOT_LIMITS_CACHE = _plot_limits(F)
        _save_plot_limits(_PLOT_LIMITS_CACHE)
        logging.info(f"[plot] fixed color limits from initial slice: {path}")
        logging.info(f"[plot] color limits: {_PLOT_LIMITS_CACHE}")
        return _PLOT_LIMITS_CACHE
    if _PLOT_LIMITS_CACHE is None:
        _PLOT_LIMITS_CACHE = _load_plot_limits()
    if _PLOT_LIMITS_CACHE is None:
        raise RuntimeError(
            f"Fixed color-limit file is unavailable: {path}. "
            "Restore the file produced at t=0 or restart with PNG plots disabled."
        )
    return _PLOT_LIMITS_CACHE


def _in_plane_vorticity(F, plane, extent):
    da = (extent[1] - extent[0])*PC_CM/max(F["vx"].shape[0], 1)
    db = (extent[3] - extent[2])*PC_CM/max(F["vx"].shape[1], 1)
    edge_order = 2 if min(F["vx"].shape) > 2 else 1
    if plane == "xy":
        omega = np.gradient(F["vy"], da, axis=0, edge_order=edge_order) - np.gradient(F["vx"], db, axis=1, edge_order=edge_order)
        label = r"$\omega_z\,[\mathrm{Myr^{-1}}]$"
    elif plane == "xz":
        omega = np.gradient(F["vx"], db, axis=1, edge_order=edge_order) - np.gradient(F["vz"], da, axis=0, edge_order=edge_order)
        label = r"$\omega_y\,[\mathrm{Myr^{-1}}]$"
    else:
        omega = np.gradient(F["vz"], da, axis=0, edge_order=edge_order) - np.gradient(F["vy"], db, axis=1, edge_order=edge_order)
        label = r"$\omega_x\,[\mathrm{Myr^{-1}}]$"
    F["vorticity_myr"] = omega*(1.0e6*YR_S)
    F["vorticity_label"] = label

def get_slice_fields(U, plane, D_bins=None, fcl1=None, fcl2=None):
    rho, vx, vy, vz, P = prim_from_cons(U)

    Nx, Ny, Nz = rho.shape
    T = eos.temperature_from_pressure_density(P, rho)
    eint = eos.internal_energy(U)
    ekin = eos.kinetic_energy_density(U)
    etherm_frac = eint / np.maximum(np.abs(U[4]), 1.0e-30)
    ekin_over_eint = ekin / np.maximum(eint, 1.0e-30)
    Dtot = None
    if (D_bins is not None):
        Dtot = np.sum(np.clip(D_bins,0,None), axis=-1)

    xmin, xmax = _axis_bounds("x")
    ymin, ymax = _axis_bounds("y")
    zmin, zmax = _axis_bounds("z")
    x = _extent_1d(xmin, xmax, Nx)
    y = _extent_1d(ymin, ymax, Ny)
    z = _extent_1d(zmin, zmax, Nz)

    if plane.lower()=="xy":
        k = Nz//2
        F = {"rho": rho[:,:,k], "T": T[:,:,k], "P": P[:,:,k],
             "vx": vx[:,:,k], "vy": vy[:,:,k], "vz": vz[:,:,k],
             "eint": eint[:,:,k], "ekin": ekin[:,:,k], "etherm_frac": etherm_frac[:,:,k], "ekin_over_eint": ekin_over_eint[:,:,k],
             "D": (Dtot[:,:,k] if Dtot is not None else None),
             "Dbins": (D_bins[:,:,k,:] if D_bins is not None else None),
             "fcl1": (fcl1[:,:,k] if fcl1 is not None else None),
             "fcl2": (fcl2[:,:,k] if fcl2 is not None else None)}
        extent = [xmin, xmax, ymin, ymax]
        axes   = ("X [pc]","Y [pc]")
        ftype  = "xy"
    elif plane.lower()=="xz":
        j = Ny//2
        F = {"rho": rho[:,j,:], "T": T[:,j,:], "P": P[:,j,:],
             "vx": vx[:,j,:], "vy": vy[:,j,:], "vz": vz[:,j,:],
             "eint": eint[:,j,:], "ekin": ekin[:,j,:], "etherm_frac": etherm_frac[:,j,:], "ekin_over_eint": ekin_over_eint[:,j,:],
             "D": (Dtot[:,j,:] if Dtot is not None else None),
             "Dbins": (D_bins[:,j,:,:] if D_bins is not None else None),
             "fcl1": (fcl1[:,j,:] if fcl1 is not None else None),
             "fcl2": (fcl2[:,j,:] if fcl2 is not None else None)}
        extent = [xmin, xmax, zmin, zmax]
        axes   = ("X [pc]","Z [pc]")
        ftype  = "xz"
    else: # "yz"
        i = Nx//2
        F = {"rho": rho[i,:,:], "T": T[i,:,:], "P": P[i,:,:],
             "vx": vx[i,:,:], "vy": vy[i,:,:], "vz": vz[i,:,:],
             "eint": eint[i,:,:], "ekin": ekin[i,:,:], "etherm_frac": etherm_frac[i,:,:], "ekin_over_eint": ekin_over_eint[i,:,:],
             "D": (Dtot[i,:,:] if Dtot is not None else None),
             "Dbins": (D_bins[i,:,:,:] if D_bins is not None else None),
             "fcl1": (fcl1[i,:,:] if fcl1 is not None else None),
             "fcl2": (fcl2[i,:,:] if fcl2 is not None else None)}
        extent = [ymin, ymax, zmin, zmax]
        axes   = ("Y [pc]","Z [pc]")
        ftype  = "yz"

    _in_plane_vorticity(F, ftype, extent)
    return F, extent, axes, ftype


def get_slice_fields_from_2d(U2, plane, D2=None, fcl1=None, fcl2=None):
    """Build plotting fields from an already gathered central 2-D conserved slice."""
    plane = plane.lower()
    rho, vx, vy, vz, P = prim_from_cons(U2)
    T = eos.temperature_from_pressure_density(P, rho)
    eint = eos.internal_energy(U2)
    ekin = eos.kinetic_energy_density(U2)
    etherm_frac = eint / np.maximum(np.abs(U2[4]), 1.0e-30)
    ekin_over_eint = ekin / np.maximum(eint, 1.0e-30)
    Dtot = np.sum(np.clip(D2, 0, None), axis=-1) if D2 is not None else None

    xmin, xmax = _axis_bounds("x")
    ymin, ymax = _axis_bounds("y")
    zmin, zmax = _axis_bounds("z")
    if plane == "xy":
        extent = [xmin, xmax, ymin, ymax]
        axes = ("X [pc]", "Y [pc]")
    elif plane == "xz":
        extent = [xmin, xmax, zmin, zmax]
        axes = ("X [pc]", "Z [pc]")
    else:
        extent = [ymin, ymax, zmin, zmax]
        axes = ("Y [pc]", "Z [pc]")

    F = {
        "rho": rho, "T": T, "P": P,
        "vx": vx, "vy": vy, "vz": vz,
        "eint": eint, "ekin": ekin,
        "etherm_frac": etherm_frac,
        "ekin_over_eint": ekin_over_eint,
        "D": Dtot,
        "Dbins": D2,
        "fcl1": fcl1,
        "fcl2": fcl2,
    }
    _in_plane_vorticity(F, plane, extent)
    return F, extent, axes, plane

# ------------------- unified 3D HDF5 writer -------------------

def write_h5_3d(U, D_bins, fcl1, fcl2, t_sec, step, snap_id, kind="std"):
    tag = BASEN
    if kind == "chk":
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_chk_{snap_id:04d}")
    elif kind == "plt":
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_plt_{snap_id:04d}")
    else:
        fname = os.path.join(OUTDIR, f"3D_{tag}_hdf5_{snap_id:04d}")
    if os.path.exists(fname):
        return

    rho, vx, vy, vz, P = prim_from_cons(U)
    T = eos.temperature_from_pressure_density(P, rho)
    eint = eos.internal_energy(U)
    ekin = eos.kinetic_energy_density(U)
    etherm_frac = eint / np.maximum(np.abs(U[4]), 1.0e-30)
    ekin_over_eint = ekin / np.maximum(eint, 1.0e-30)

    def _write(tmp_path):
        with h5py.File(tmp_path, "w") as h:
            _h5_stream_write(h, "time", np.array(t_sec))
            _h5_stream_write(h, "nstep", np.array(step, dtype=np.int64))
            _h5_stream_write(h, "snap", np.array(snap_id, dtype=np.int64))
            _h5_stream_write(h, "dimensionality", np.array(3 if DIM3D else 2, dtype=np.int32))
            _h5_stream_write(h, "bbox", np.array([_axis_bounds("x"), _axis_bounds("y"), _axis_bounds("z")], dtype=np.float32))
            _h5_stream_write(h, "cell_sizes", np.array([dx,dy,dz], dtype=np.float32))
            _h5_stream_write(h, "frame_offset_cm", np.array(float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)), dtype=np.float64))
            _h5_stream_write(h, "frame_remainder_cm", np.array(float(globals().get("FRAME_REMAINDER_CM_RUNTIME", 0.0)), dtype=np.float64))
            h.attrs["frame_axis"] = str(globals().get("FRAME_AXIS_RUNTIME", globals().get("MOVING_FRAME_AXIS", "z")))

            _h5_stream_write(h, "U0", U[0])
            _h5_stream_write(h, "U1", U[1])
            _h5_stream_write(h, "U2", U[2])
            _h5_stream_write(h, "U3", U[3])
            _h5_stream_write(h, "U4", U[4])
            if U.shape[0] > 5: _h5_stream_write(h, "U5", U[5])

            _h5_stream_write(h, "dens", rho)
            _h5_stream_write(h, "pres", P)
            _h5_stream_write(h, "temp", T)
            _h5_stream_write(h, "eint", eint)
            _h5_stream_write(h, "ekin", ekin)
            _h5_stream_write(h, "etherm_frac", etherm_frac)
            _h5_stream_write(h, "ekin_over_eint", ekin_over_eint)
            _h5_stream_write(h, "velx", vx)
            _h5_stream_write(h, "vely", vy)
            _h5_stream_write(h, "velz", vz)

            if fcl1 is not None:
                _h5_stream_write(h, "fcl1", fcl1)
            if fcl2 is not None:
                _h5_stream_write(h, "fcl2", fcl2)

            if D_bins is not None:
                NB = D_bins.shape[-1]
                for i in range(NB):
                    _h5_stream_write(h, f"dst{i}", D_bins[..., i])

    ok = _atomic_write(fname, _write)
    if ok:
        kind_label = {"std":"[hdf5]  ", "plt":"[plt]", "chk":"[chk]"}[kind]
        logging.info(f"{kind_label} wrote {fname}")


def make_energy_diagnostic_png(F, extent, axes, t_yr, snap_id, limits):
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    panels = [
        ("eint", "logeint", r"$\log e_{\rm int}\,[\mathrm{erg\,cm^{-3}}]$"),
        ("ekin", "logekin", r"$\log e_{\rm kin}\,[\mathrm{erg\,cm^{-3}}]$"),
        ("vorticity_myr", "vorticity", F["vorticity_label"]),
        ("etherm_frac", "logetherm", r"$\log(e_{\rm int}/E)$"),
    ]
    for ax, (field, key, label) in zip(axs.ravel(), panels):
        if field == "vorticity_myr":
            A = F[field].T
            cmap = "RdBu_r"
        else:
            floor = 10.0**max(limits[key][0], -300.0)
            A = np.log10(np.clip(F[field], floor, None)).T
            cmap = "viridis"
        im = ax.imshow(
            A, origin="lower", extent=extent,
            vmin=limits[key][0], vmax=limits[key][1], cmap=cmap, aspect="equal",
        )
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.82)
    axs[0, 0].set_ylabel(axes[1])
    axs[1, 0].set_ylabel(axes[1])
    axs[1, 0].set_xlabel(axes[0])
    axs[1, 1].set_xlabel(axes[0])
    fig.suptitle(f"t = {t_yr/1e3:.2f} kyr", y=0.98)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f"diag_energy_{BASEN}_{snap_id:04d}.png")
    fig.savefig(png, dpi=FIG_DPI)
    plt.close(fig)
    logging.info(f"[diag] wrote {png}  t={t_yr/1e3:6.2f} kyr")


def _render_slice_outputs(F, extent, axes, ftype, t_yr, snap_id, nstep, initialize_plot_limits=False):
    snap_on = bool(globals().get("SNAP_PLOTS_ON", True))
    diag_on = bool(globals().get("DIAG_PLOTS_ON", True))
    limits = _fixed_plot_limits(F, initialize=initialize_plot_limits) if (snap_on or diag_on) else None

    if snap_on:
        fig = plt.figure(figsize=(13.0, 8.0))
        gs = fig.add_gridspec(
            2, 3, width_ratios=[1, 1, 0.9], height_ratios=[1, 1],
            wspace=0.5, hspace=0.0005,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax, sharey=ax)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax, sharey=ax)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)

        rho_floor = 10.0**max(limits["logrho"][0], -300.0)
        im = ax.imshow(
            np.log10(np.clip(F["rho"], rho_floor, None)).T,
            origin="lower", extent=extent, vmin=limits["logrho"][0],
            vmax=limits["logrho"][1], cmap="jet", aspect="equal",
        )
        _inset_colorbar(fig, ax, im, r"$\log\rho\,[\mathrm{g\,cm^{-3}}]$")

        T_floor = 10.0**max(limits["logT"][0], -300.0)
        im2 = ax2.imshow(
            np.log10(np.clip(F["T"], T_floor, None)).T,
            origin="lower", extent=extent, vmin=limits["logT"][0],
            vmax=limits["logT"][1], cmap="jet", aspect="equal",
        )
        _inset_colorbar(fig, ax2, im2, r"$\log\,T\,[\mathrm{K}]$")

        if F["D"] is not None:
            D_floor = 10.0**max(limits["logD"][0], -300.0)
            im3 = ax3.imshow(
                np.log10(np.clip(F["D"], D_floor, None)).T,
                origin="lower", extent=extent, cmap="jet", aspect="equal",
                vmin=limits["logD"][0], vmax=limits["logD"][1],
            )
            _inset_colorbar(fig, ax3, im3, r"$\log\,\mathcal{D}$")
        else:
            ax3.text(0.5, 0.5, "Dust OFF", ha="center", va="center", transform=ax3.transAxes)

        vmag = np.sqrt(F["vx"]**2 + F["vy"]**2 + F["vz"]**2) / 1.0e5
        im4 = ax4.imshow(
            vmag.T, origin="lower", extent=extent, cmap="jet", aspect="equal",
            vmin=limits["vmag"][0], vmax=limits["vmag"][1],
        )
        _inset_colorbar(fig, ax4, im4, r"$|{\bf v}|\,[\mathrm{km\,s^{-1}}]$")

        for axis in (ax, ax2):
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)
        for axis in (ax2, ax4):
            axis.set_ylabel("")
            axis.tick_params(labelleft=False)
        ax.set_ylabel(axes[1])
        ax3.set_ylabel(axes[1])
        ax3.set_xlabel(axes[0])
        ax4.set_xlabel(axes[0])
        fig.suptitle(f"t = {t_yr/1e3:.2f} kyr  |  {ftype.upper()} cut", y=0.985, fontsize=16)
        axm = fig.add_subplot(gs[:, 2])
        axm.axis("off")
        png = os.path.join(OUTDIR, f"snap_{BASEN}_{snap_id:04d}.png")
        fig.savefig(png, dpi=FIG_DPI)
        plt.close(fig)
        logging.info(f"[snap] wrote {png}  t={t_yr/1e3:6.2f} kyr")

    if diag_on:
        make_energy_diagnostic_png(F, extent, axes, t_yr, snap_id, limits)

    base = f"asc-{ftype}-{BASEN}_{snap_id:04d}"
    fname = os.path.join(OUTDIR, base)
    write_ascii_slice(fname, F, extent, ftype, t_yr * YR_S, nstep)
    logging.info(f"[ascii]  wrote {fname}")


def make_png_from_slice(U2, plane, t_yr, snap_id, nstep, D2=None, fcl1=None, fcl2=None, initialize_plot_limits=False):
    """Write PNG diagnostics and ASCII from a gathered 2-D central slice."""
    F, extent, axes, ftype = get_slice_fields_from_2d(U2, plane, D2, fcl1, fcl2)
    _render_slice_outputs(F, extent, axes, ftype, t_yr, snap_id, nstep, initialize_plot_limits)


def make_png(U, t_yr, snap_id, nstep, D_bins, fcl1, fcl2, initialize_plot_limits=False):
    """Serial/full-domain compatibility path."""
    if os.path.exists(os.path.join(OUTDIR, f"3D_{BASEN}_hdf5_{snap_id:04d}")):
        return
    F, extent, axes, ftype = get_slice_fields(U, VIEW_PLANE, D_bins, fcl1, fcl2)
    _render_slice_outputs(F, extent, axes, ftype, t_yr, snap_id, nstep, initialize_plot_limits)
    write_h5_3d(U, D_bins, fcl1, fcl2, t_yr * YR_S, step=nstep, snap_id=snap_id, kind="std")


def write_ascii_slice(fname, F, extent, ftype, t_sec, step):
    x0, x1, y0, y1 = extent
    nx, ny = F["rho"].shape
    xs = _extent_1d(x0, x1, nx)
    ys = _extent_1d(y0, y1, ny)
    Xp, Yp = np.meshgrid(xs, ys, indexing="ij")

    rho2 = F["rho"]; P2 = F["P"]; T2 = F["T"]
    vx2, vy2, vz2 = F["vx"], F["vy"], F["vz"]

    cols = [Xp.ravel(), Yp.ravel(),
            rho2.ravel(), P2.ravel(), T2.ravel(),
            vx2.ravel(), vy2.ravel(), vz2.ravel()]

    if (F.get("Dbins", None) is not None):
        for i in range(F["Dbins"].shape[-1]):
            cols.append( F["Dbins"][:,:,i].ravel() )

    M = np.vstack(cols).T
    with open(fname, "w") as f:
        f.write(f"# time = {t_sec:.8e} nstep = {step}\n")
        f.write(f"# plane = {ftype}   shape=({nx},{ny})\n")
        headers = ["x","y","dens","pres","temp","velx","vely","velz"]
        if (F.get("Dbins", None) is not None):
            headers += [f"dst{i}" for i in range(F['Dbins'].shape[-1])]
        f.write("# " + "  ".join([f"{i+1:<12d}" for i in range(len(headers))]) + "\n")
        f.write("# " + "  ".join([f"{h:<12s}" for h in headers]) + "\n")
        np.savetxt(f, M, fmt="%.8e")
