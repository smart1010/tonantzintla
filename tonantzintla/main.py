"""TONANTZINTLA hydrodynamics driver.

Author: Sergio Martínez González (INAOE).
Scientific advisor and feature design contributor: Guillermo Tenorio-Tagle (INAOE).
"""

import os, sys, logging, glob, time

CODE_VERSION = "TONANTZINTLA"

import numpy as np
import h5py
from . import read_data
from . import eos
from .mpi_utils import COMM, RANK, SIZE, decompose_1d, exchange_ghosts_x, exchange_ghosts_x_scalar, gather_global_x, gather_scalar_x, gather_bins_x
from ._mass_io import ensure_mass_header, compute_phase_masses_local, reduce6, append_mass_row_once, snap_exists, should_write_mass, last_time_kyr

# Keep non-root stdout quiet while preserving tracebacks on stderr.
try:
    if RANK != 0:
        sys.stdout = open(os.devnull, 'w')
except Exception:
    pass


# Rank-safe logger initialization.
_LOG_CFGD = False
def _setup_logging_once(OUTDIR, BASEN, LOGFILE=None, RESTARTING=False):
    global _LOG_CFGD
    if _LOG_CFGD:
        return
    logger = logging.getLogger("")
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # file handler only on rank 0
    if RANK == 0:
        lfn  = LOGFILE or os.path.join(OUTDIR, f"{BASEN}.log")
        mode = 'a' if RESTARTING else 'w'
        try:
            fh = logging.FileHandler(lfn, mode=mode, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass

    # console: quiet non-root ranks to avoid interleaving

    _LOG_CFGD = True
    if RANK == 0:
        logging.info('===== START (rank 0) =====')
# --- end logger setup ---



def _install_settings_source(source):
    """Install runtime settings without importing the generated file."""
    import types

    module_name = f"{__package__}.settings"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__file__ = os.path.join(os.path.dirname(__file__), "settings.py")
        module.__package__ = __package__
        sys.modules[module_name] = module
        package = sys.modules.get(__package__)
        if package is not None:
            setattr(package, "settings", module)
    else:
        for name in tuple(vars(module)):
            if not name.startswith("__"):
                vars(module).pop(name, None)

    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


# --- MPI master I/O gate (root decides; all ranks follow) ---


# --- Snapshot scheduling (rank 0, accumulator, single writer) ---




class _DummyComm:
    def Barrier(self):
        return None


class SnapshotManager:
    """Simulation-time snapshot scheduler with restart-aware phase."""
    def __init__(self, comm, rank, interval_sec, phase_sec=0.0):
        self.comm = comm if comm is not None else _DummyComm()
        self.rank = int(rank) if rank is not None else 0
        self.interval = float(interval_sec)
        self.accum = (float(phase_sec) % self.interval) if self.interval > 0.0 else 0.0

    def due(self, dt):
        if self.interval <= 0.0:
            return False
        self.accum += float(dt)
        if self.accum < self.interval:
            return False
        self.accum %= self.interval
        return True

def _io_allow_this_step(COMM, RANK, t, YR_S, MASS_FILE):
    try:
        t_kyr = float(t) / float(YR_S) / 1.0e3
        if RANK == 0:
            last = last_time_kyr(MASS_FILE)
            # Allow if mass log has no entries yet (first write), or time advanced by > eps
            allow = (last is None) or (t_kyr > (last + 1e-15)) or (abs(t_kyr) < 1e-20)
        else:
            allow = None
        allow = COMM.bcast(allow, root=0)
        return bool(allow)
    except Exception:
        # Fail-open: do not block writes if something goes wrong in the gate
        return True

def _consume(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def _kyr(t_sec):
    return t_sec / (float(globals().get("YR_S", 3.15576e7)) * 1.0e3)

def _log_snapshot_messages(snap_id, t_sec):
    """Log snapshot output paths and simulation time."""
    outdir = globals().get("OUTDIR", ".")
    basen  = globals().get("BASEN", "clouds")
    png   = os.path.join(outdir, f"snap_{basen}_{snap_id:04d}.png")
    asc   = os.path.join(outdir, f"asc-xy-{basen}_{snap_id:04d}")
    h5plt = os.path.join(outdir, f"3D_{basen}_hdf5_{snap_id:04d}")
    logging.info(f"[snap]  wrote {png}  t={_kyr(t_sec):6.2f} kyr")
    logging.info(f"[ascii] wrote {asc}")
    logging.info(f"[hdf5]  wrote {h5plt}")

def log_header():
    # Print domain extents in cm (not pc)
    pc2cm = float(globals().get("PC_CM", 3.0856775814913673e18))
    xmin_cm = float(globals().get("XMIN_PC", 0.0)) * pc2cm
    xmax_cm = float(globals().get("XMAX_PC", 0.0)) * pc2cm
    ymin_cm = float(globals().get("YMIN_PC", 0.0)) * pc2cm
    ymax_cm = float(globals().get("YMAX_PC", 0.0)) * pc2cm
    zmin_cm = float(globals().get("ZMIN_PC", 0.0)) * pc2cm
    zmax_cm = float(globals().get("ZMAX_PC", 0.0)) * pc2cm

    logging.info(f"Code: {CODE_VERSION}")
    logging.info("==============================================================")
    from . import settings as _S
    _RIEM, _ORD = _S.RIEMANN, _S.ORDER
    _time_scheme = "MUSCL-Hancock" if str(_ORD).lower() == "plm" else "Godunov-Euler"
    logging.info(f"= {str(_RIEM).upper()} hydro ({_time_scheme}) + tracers + dust (cinder)             =")
    logging.info("==============================================================")
    logging.info(f"Grid: {Nx}x{Ny}x{Nz}  Domain [cm]: "
                 f"[{xmin_cm:.6e},{xmax_cm:.6e}] × "
                 f"[{ymin_cm:.6e},{ymax_cm:.6e}] × "
                 f"[{zmin_cm:.6e},{zmax_cm:.6e}] | DIM3D={DIM3D}")
    logging.info(f"Cooling={COOLING_ON}  Dust={DUST_ON}  Tracers={TRACERS_ON}  View={VIEW_PLANE}")
    logging.info(f"Plots: snap={SNAP_PLOTS_ON}  diag={DIAG_PLOTS_ON}  color limits=initial slice")
    if str(_RIEM).lower() == "hllc":
        logging.info(f"Carbuncle stabilization={getattr(_S, 'CARBUNCLE_FIX_ON', True)}  "
                     f"p-threshold={getattr(_S, 'SHOCK_PRESSURE_THRESHOLD', 0.2)}  "
                     f"p-full={getattr(_S, 'SHOCK_FULL_HLLE_THRESHOLD', 0.6)}  "
                     f"compression-threshold={getattr(_S, 'SHOCK_COMPRESSION_THRESHOLD', 0.05)}")
    logging.info(f"Turbulence: ON={TURB_ON}  sigmas[km/s]: amb={SIGMA_T_AMB_KMS}  c1={SIGMA_T_C1_KMS}  c2={SIGMA_T_C2_KMS}")

# ----------------------- tolerant checkpoint I/O -----------------------

def _load_checkpoint(path):
    """
    Load checkpoint/plotfile. Accepts 'rho' or 'dens'.
    Missing momentum or energy fields are initialized using conservative fallback values.
    Returns: U, D_bins, f1, f2, t_sec, step, chk_id, snap_next
    """
    with h5py.File(path, "r") as h:
        # density key: 'rho' or 'dens'
        if "rho" in h:
            k_rho = "rho"
        elif "dens" in h:
            k_rho = "dens"
        else:
            raise KeyError("Unable to open object (neither 'rho' nor 'dens' found)")

        Nx_, Ny_, Nz_ = h[k_rho].shape
        U = np.zeros((6, Nx_, Ny_, Nz_), dtype=np.float64)
        U[0] = h[k_rho][...]

        # momentum/energy keys (optional but preferred)
        k_mx = "mx" if "mx" in h else None
        k_my = "my" if "my" in h else None
        k_mz = "mz" if "mz" in h else None
        k_E  = "E"  if "E"  in h else None
        k_ei = "rho_eint" if "rho_eint" in h else ("U5" if "U5" in h else ("eint" if "eint" in h else None))

        have_mom = (k_mx is not None) and (k_my is not None) and (k_mz is not None)
        have_E   = (k_E  is not None)

        if have_mom:
            U[1] = h[k_mx][...]
            U[2] = h[k_my][...]
            U[3] = h[k_mz][...]
        else:
            U[1] = 0.0; U[2] = 0.0; U[3] = 0.0

        if have_E:
            U[4] = h[k_E][...]
        else:
            T_fill = globals().get("RESTART_T_K", globals().get("T_AMB_K", 1.0e4))
            rho = U[0]
            P = eos.pressure_from_density_temperature(rho, T_fill)
            U[4] = P/(5.0/3.0 - 1.0) + eos.kinetic_energy_density(U)
        U[5] = h[k_ei][...] if k_ei is not None else eos.internal_energy_from_total(U)
        U = eos.reconcile_dual_energy(U)

        # tracers
        f1 = h["f1"][...] if "f1" in h else (h["tr1"][...] if "tr1" in h else np.zeros((Nx_,Ny_,Nz_)))
        f2 = h["f2"][...] if "f2" in h else (h["tr2"][...] if "tr2" in h else np.zeros((Nx_,Ny_,Nz_)))

        # dust bins: dst*
        dust_keys = sorted([k for k in h.keys() if k.startswith("dst")])
        D_bins = None
        if dust_keys:
            NB = len(dust_keys)
            D_bins = np.zeros((Nx_, Ny_, Nz_, NB), dtype=np.float32)
            for i,k in enumerate(dust_keys):
                D_bins[..., i] = h[k][...]
        else:
            NB = int(h.attrs.get("NBINS", 0))

        # small helper
        def _get_scalar(name, default=None):
            if name in h:
                try: return h[name][()].item() if hasattr(h[name][()], "item") else type(default or 0)(h[name][()])
                except Exception: pass
            if name in h.attrs:
                try: 
                    v = h.attrs[name]
                    return v.item() if hasattr(v, "item") else v
                except Exception: 
                    pass
            return default

        # metadata
        t_sec     = float(_get_scalar("time", 0.0))
        step      = int(_get_scalar("nstep", 0))

        # Restart counters
        chk_id    = _get_scalar("chk", None)
        snap_next = _get_scalar("snap", None)

        # Restart metadata fallback
        if chk_id is None and snap_next is not None:
            try:
                chk_id = int(snap_next)
                snap_next = int(snap_next)
            except Exception:
                chk_id = 0
                snap_next = 0
        if chk_id is None:
            # Infer the checkpoint index from the filename when metadata is absent.
            base = os.path.basename(path)
            try:
                chk_id = int(base.split("_")[-1])
            except Exception:
                chk_id = 0
        if snap_next is None:
            snap_next = 0
        frame_offset = float(_get_scalar("frame_offset_cm", 0.0))
        frame_remainder = float(_get_scalar("frame_remainder_cm", 0.0))

    return U, D_bins, f1, f2, t_sec, step, int(chk_id), int(snap_next), frame_offset, frame_remainder

def _file_suffix_id(path):
    try:
        return int(os.path.basename(path).split("_")[-1])
    except Exception:
        return -1

def _find_latest_checkpoint(outdir, basen):
    if not os.path.isdir(outdir):
        return None
    preferred = [os.path.join(outdir, n) for n in os.listdir(outdir)
                 if n.startswith(f"3D_{basen}_hdf5_chk_") and os.path.isfile(os.path.join(outdir, n))]
    if preferred:
        return max(preferred, key=_file_suffix_id)
    allchk = [os.path.join(outdir, n) for n in os.listdir(outdir)
              if ("chk" in n) and os.path.isfile(os.path.join(outdir, n))]
    return max(allchk, key=_file_suffix_id) if allchk else None

def _next_standard_snap_id(outdir, basen):
    if not os.path.isdir(outdir):
        return 0
    prefix = f"3D_{basen}_hdf5_"
    ids = []
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if not (name.startswith(prefix) and os.path.isfile(path)):
            continue
        tail = name[len(prefix):]
        if tail.isdigit() and os.path.getsize(path) > 0:
            ids.append(int(tail))
    return (max(ids) + 1) if ids else 0


def _sorted_dust_keys(keys):
    def _num(k):
        try:
            return int(str(k)[3:])
        except Exception:
            return 10**9
    return sorted((k for k in keys if str(k).startswith("dst")), key=_num)


def _load_checkpoint_local(path, i0, i1, comm, rank):
    """Read only this rank's X slab from a checkpoint."""
    if rank == 0:
        with h5py.File(path, "r") as h:
            keys = set(h.keys())
            krho = "rho" if "rho" in keys else ("dens" if "dens" in keys else ("U0" if "U0" in keys else None))
            if krho is None:
                raise KeyError("checkpoint has no rho/dens/U0 dataset")
            shape = tuple(h[krho].shape)
            meta = {
                "shape": shape,
                "krho": krho,
                "kmx": "mx" if "mx" in keys else ("U1" if "U1" in keys else None),
                "kmy": "my" if "my" in keys else ("U2" if "U2" in keys else None),
                "kmz": "mz" if "mz" in keys else ("U3" if "U3" in keys else None),
                "kE": "E" if "E" in keys else ("U4" if "U4" in keys else None),
                "kei": "rho_eint" if "rho_eint" in keys else ("U5" if "U5" in keys else ("eint" if "eint" in keys else None)),
                "kf1": "f1" if "f1" in keys else ("fcl1" if "fcl1" in keys else ("tr1" if "tr1" in keys else None)),
                "kf2": "f2" if "f2" in keys else ("fcl2" if "fcl2" in keys else ("tr2" if "tr2" in keys else None)),
                "dust": _sorted_dust_keys(keys),
            }
            def scalar(name, default):
                if name in h:
                    v = h[name][()]
                    return v.item() if hasattr(v, "item") else v
                if name in h.attrs:
                    v = h.attrs[name]
                    return v.item() if hasattr(v, "item") else v
                return default
            meta.update({
                "time": float(scalar("time", 0.0)),
                "step": int(scalar("nstep", 0)),
                "chk": scalar("chk", None),
                "snap": int(scalar("snap", 0)),
                "frame_offset": float(scalar("frame_offset_cm", 0.0)),
                "frame_remainder": float(scalar("frame_remainder_cm", 0.0)),
            })
            if meta["chk"] is None:
                try:
                    meta["chk"] = int(os.path.basename(path).split("_")[-1])
                except Exception:
                    meta["chk"] = 0
    else:
        meta = None
    meta = comm.bcast(meta, root=0)

    nxg, nyg, nzg = meta["shape"]
    if (nxg, nyg, nzg) != (Nx, Ny, Nz):
        raise ValueError(f"checkpoint grid {meta['shape']} does not match configured grid {(Nx, Ny, Nz)}")

    sl = np.s_[int(i0):int(i1), :, :]
    nxl = int(i1) - int(i0)
    U = np.zeros((6, nxl, Ny, Nz), dtype=np.float32)
    with h5py.File(path, "r") as h:
        U[0] = h[meta["krho"]][sl]
        for n, key in ((1, meta["kmx"]), (2, meta["kmy"]), (3, meta["kmz"])):
            if key is not None:
                U[n] = h[key][sl]
        if meta["kE"] is not None:
            U[4] = h[meta["kE"]][sl]
        else:
            T_fill = globals().get("RESTART_T_K", globals().get("T_AMB", 1.0e4))
            P = eos.pressure_from_density_temperature(U[0], T_fill)
            U[4] = P / ((5.0 / 3.0) - 1.0) + eos.kinetic_energy_density(U)
        if meta["kei"] is not None:
            U[5] = h[meta["kei"]][sl]
        else:
            U[5] = eos.internal_energy_from_total(U)

        f1 = h[meta["kf1"]][sl].astype(np.float32, copy=False) if meta["kf1"] else np.zeros((nxl, Ny, Nz), dtype=np.float32)
        f2 = h[meta["kf2"]][sl].astype(np.float32, copy=False) if meta["kf2"] else np.zeros((nxl, Ny, Nz), dtype=np.float32)
        if meta["dust"]:
            D = np.empty((nxl, Ny, Nz, len(meta["dust"])), dtype=np.float32)
            for ib, key in enumerate(meta["dust"]):
                D[..., ib] = h[key][sl]
        else:
            D = None

    U = eos.reconcile_dual_energy(U)
    return (U, D, f1, f2, meta["time"], meta["step"], int(meta["chk"]),
            int(meta["snap"]), meta["frame_offset"], meta["frame_remainder"])


def _gather_view_slice(U, D_bins, f1, f2, nx_local, i0, i1, starts, counts, comm, rank):
    """Collect the selected central 2-D plane on rank 0 with explicit buffers.

    mpi4py object ``gather`` is deliberately avoided here.  Every non-root rank
    sends only its local slice, and rank 0 receives directly into the correct
    global X interval.  This keeps the transfer small and deterministic for
    large 3-D runs.
    """
    plane = str(VIEW_PLANE).lower()
    Ui = U[:, 1:1 + nx_local, :, :]
    Di = D_bins[1:1 + nx_local, :, :, :] if D_bins is not None else None
    f1i = f1[1:1 + nx_local, :, :] if f1 is not None else None
    f2i = f2[1:1 + nx_local, :, :] if f2 is not None else None

    has_d = Di is not None
    has_f1 = f1i is not None
    has_f2 = f2i is not None
    nbins = int(Di.shape[-1]) if has_d else 0

    # Tags are safe because snapshot writes are serialized by the caller.
    TAG_U, TAG_D, TAG_F1, TAG_F2 = 640, 641, 642, 643

    if rank == 0:
        logging.info(f"[slice] begin plane={plane}")

    if plane in ("xz", "xy"):
        if plane == "xz":
            j = Ny // 2
            u_local = np.ascontiguousarray(Ui[:, :, j, :])
            d_local = np.ascontiguousarray(Di[:, j, :, :]) if has_d else None
            f1_local = np.ascontiguousarray(f1i[:, j, :]) if has_f1 else None
            f2_local = np.ascontiguousarray(f2i[:, j, :]) if has_f2 else None
            u_shape = (6, Nx, Nz)
            d_shape = (Nx, Nz, nbins)
            f_shape = (Nx, Nz)
        else:
            k = Nz // 2
            u_local = np.ascontiguousarray(Ui[:, :, :, k])
            d_local = np.ascontiguousarray(Di[:, :, k, :]) if has_d else None
            f1_local = np.ascontiguousarray(f1i[:, :, k]) if has_f1 else None
            f2_local = np.ascontiguousarray(f2i[:, :, k]) if has_f2 else None
            u_shape = (6, Nx, Ny)
            d_shape = (Nx, Ny, nbins)
            f_shape = (Nx, Ny)

        if rank == 0:
            U2 = np.empty(u_shape, dtype=u_local.dtype)
            D2 = np.empty(d_shape, dtype=d_local.dtype) if has_d else None
            F1 = np.empty(f_shape, dtype=f1_local.dtype) if has_f1 else None
            F2 = np.empty(f_shape, dtype=f2_local.dtype) if has_f2 else None

            for r, (a, n) in enumerate(zip(starts, counts)):
                a = int(a)
                b = a + int(n)
                if r == 0:
                    ur = u_local
                    dr = d_local
                    f1r = f1_local
                    f2r = f2_local
                else:
                    trailing = Nz if plane == "xz" else Ny
                    ur = np.empty((6, int(n), trailing), dtype=u_local.dtype)
                    comm.Recv(ur, source=r, tag=TAG_U)
                    dr = np.empty((int(n), trailing, nbins), dtype=d_local.dtype) if has_d else None
                    f1r = np.empty((int(n), trailing), dtype=f1_local.dtype) if has_f1 else None
                    f2r = np.empty((int(n), trailing), dtype=f2_local.dtype) if has_f2 else None
                    if has_d:
                        comm.Recv(dr, source=r, tag=TAG_D)
                    if has_f1:
                        comm.Recv(f1r, source=r, tag=TAG_F1)
                    if has_f2:
                        comm.Recv(f2r, source=r, tag=TAG_F2)

                U2[:, a:b, :] = ur
                if has_d:
                    D2[a:b, :, :] = dr
                if has_f1:
                    F1[a:b, :] = f1r
                if has_f2:
                    F2[a:b, :] = f2r

            logging.info(f"[slice] complete plane={plane}")
            return U2, D2, F1, F2

        comm.Send(u_local, dest=0, tag=TAG_U)
        if has_d:
            comm.Send(d_local, dest=0, tag=TAG_D)
        if has_f1:
            comm.Send(f1_local, dest=0, tag=TAG_F1)
        if has_f2:
            comm.Send(f2_local, dest=0, tag=TAG_F2)
        return None, None, None, None

    # YZ plane belongs to exactly one X slab, so only the owner sends it.
    ig = Nx // 2
    owner = next(r for r, (a, n) in enumerate(zip(starts, counts))
                 if int(a) <= ig < int(a) + int(n))

    if rank == owner:
        il = ig - int(i0)
        u_local = np.ascontiguousarray(Ui[:, il, :, :])
        d_local = np.ascontiguousarray(Di[il, :, :, :]) if has_d else None
        f1_local = np.ascontiguousarray(f1i[il, :, :]) if has_f1 else None
        f2_local = np.ascontiguousarray(f2i[il, :, :]) if has_f2 else None
    else:
        u_local = d_local = f1_local = f2_local = None

    if rank == 0:
        if owner == 0:
            U2, D2, F1, F2 = u_local, d_local, f1_local, f2_local
        else:
            U2 = np.empty((6, Ny, Nz), dtype=Ui.dtype)
            comm.Recv(U2, source=owner, tag=TAG_U)
            D2 = np.empty((Ny, Nz, nbins), dtype=Di.dtype) if has_d else None
            F1 = np.empty((Ny, Nz), dtype=f1i.dtype) if has_f1 else None
            F2 = np.empty((Ny, Nz), dtype=f2i.dtype) if has_f2 else None
            if has_d:
                comm.Recv(D2, source=owner, tag=TAG_D)
            if has_f1:
                comm.Recv(F1, source=owner, tag=TAG_F1)
            if has_f2:
                comm.Recv(F2, source=owner, tag=TAG_F2)
        logging.info(f"[slice] complete plane={plane}")
        return U2, D2, F1, F2

    if rank == owner:
        comm.Send(u_local, dest=0, tag=TAG_U)
        if has_d:
            comm.Send(d_local, dest=0, tag=TAG_D)
        if has_f1:
            comm.Send(f1_local, dest=0, tag=TAG_F1)
        if has_f2:
            comm.Send(f2_local, dest=0, tag=TAG_F2)
    return None, None, None, None


def _distributed_h5_path(kind, out_id, outdir, basen):
    if kind == "chk":
        return os.path.join(outdir, f"3D_{basen}_hdf5_chk_{int(out_id):04d}")
    if kind == "plt":
        return os.path.join(outdir, f"3D_{basen}_hdf5_plt_{int(out_id):04d}")
    return os.path.join(outdir, f"3D_{basen}_hdf5_{int(out_id):04d}")


def _write_h5_distributed(U, D_bins, f1, f2, nx_local, starts, counts,
                          comm, rank, t, step, out_id, kind="std", snap_next=None):
    """Single HDF5 writer fed one X slab at a time; no global cube is assembled."""
    path = _distributed_h5_path(kind, out_id, OUTDIR, BASEN)
    if rank == 0:
        skip = os.path.exists(path)
    else:
        skip = None
    skip = comm.bcast(skip, root=0)
    if skip:
        comm.Barrier()
        return path

    has_f1 = bool(comm.allreduce(1 if f1 is not None else 0))
    has_f2 = bool(comm.allreduce(1 if f2 is not None else 0))
    has_d = bool(comm.allreduce(1 if D_bins is not None else 0))
    nbins_local = int(D_bins.shape[-1]) if D_bins is not None else 0
    nbins = int(comm.bcast(nbins_local if rank == 0 else None, root=0)) if has_d else 0

    Ui = np.ascontiguousarray(U[:, 1:1 + nx_local, :, :])
    F1i = np.ascontiguousarray(f1[1:1 + nx_local, :, :]) if f1 is not None else None
    F2i = np.ascontiguousarray(f2[1:1 + nx_local, :, :]) if f2 is not None else None
    Di = np.ascontiguousarray(D_bins[1:1 + nx_local, :, :, :]) if D_bins is not None else None

    if rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        if os.path.exists(tmp):
            os.remove(tmp)
        h = h5py.File(tmp, "w")
        max_nx = max(int(n) for n in counts)
        c3 = (min(max_nx, Nx), min(Ny, 64), min(Nz, 64))
        comp = {"compression": "gzip", "compression_opts": 1} if kind == "chk" else {}

        def d3(name, dtype=np.float32):
            return h.create_dataset(name, shape=(Nx, Ny, Nz), dtype=dtype, chunks=c3, **comp)

        if kind == "chk":
            datasets = {name: d3(name) for name in ("rho", "mx", "my", "mz", "E", "rho_eint")}
            h.attrs["NBINS"] = nbins
            h.create_dataset("time", data=np.array(t, dtype=np.float64))
            h.create_dataset("nstep", data=np.array(step, dtype=np.int64))
            h.create_dataset("chk", data=np.array(out_id, dtype=np.int64))
            h.create_dataset("snap", data=np.array(0 if snap_next is None else snap_next, dtype=np.int64))
            h.create_dataset("frame_offset_cm", data=np.array(float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)), dtype=np.float64))
            h.create_dataset("frame_remainder_cm", data=np.array(float(globals().get("FRAME_REMAINDER_CM_RUNTIME", 0.0)), dtype=np.float64))
            h.attrs["frame_axis"] = str(globals().get("FRAME_AXIS_RUNTIME", globals().get("MOVING_FRAME_AXIS", "z")))
            h.attrs["dx"] = float(dx); h.attrs["dy"] = float(dy); h.attrs["dz"] = float(dz)
            h.attrs["Nx"] = int(Nx); h.attrs["Ny"] = int(Ny); h.attrs["Nz"] = int(Nz)
            h.attrs["gamma"] = float(gamma)
        else:
            h.create_dataset("time", data=np.array(t, dtype=np.float64))
            h.create_dataset("nstep", data=np.array(step, dtype=np.int64))
            h.create_dataset("snap", data=np.array(out_id, dtype=np.int64))
            h.create_dataset("dimensionality", data=np.array(3 if DIM3D else 2, dtype=np.int32))
            off = float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)) / PC_CM
            ax = str(globals().get("FRAME_AXIS_RUNTIME", globals().get("MOVING_FRAME_AXIS", "z"))).lower()
            xb = [XMIN_PC + (off if ax == "x" else 0.0), XMAX_PC + (off if ax == "x" else 0.0)]
            yb = [YMIN_PC + (off if ax == "y" else 0.0), YMAX_PC + (off if ax == "y" else 0.0)]
            zb = [ZMIN_PC + (off if ax == "z" else 0.0), ZMAX_PC + (off if ax == "z" else 0.0)]
            h.create_dataset("bbox", data=np.array([xb, yb, zb], dtype=np.float32))
            h.create_dataset("cell_sizes", data=np.array([dx, dy, dz], dtype=np.float32))
            h.create_dataset("frame_offset_cm", data=np.array(float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)), dtype=np.float64))
            h.create_dataset("frame_remainder_cm", data=np.array(float(globals().get("FRAME_REMAINDER_CM_RUNTIME", 0.0)), dtype=np.float64))
            h.attrs["frame_axis"] = ax
            datasets = {name: d3(name) for name in (
                "U0", "U1", "U2", "U3", "U4", "U5", "dens", "pres", "temp",
                "eint", "ekin", "etherm_frac", "ekin_over_eint", "velx", "vely", "velz"
            )}

        f1ds = d3("f1" if kind == "chk" else "fcl1") if has_f1 else None
        f2ds = d3("f2" if kind == "chk" else "fcl2") if has_f2 else None
        dds = [d3(f"dst{ib}") for ib in range(nbins)] if has_d else []
    else:
        h = None
        tmp = None
        datasets = f1ds = f2ds = dds = None

    try:
        for r, (a, n) in enumerate(zip(starts, counts)):
            b = a + n
            if rank == r:
                if r == 0:
                    Ur, F1r, F2r, Dr = Ui, F1i, F2i, Di
                else:
                    comm.Send(Ui, dest=0, tag=500)
                    if has_f1: comm.Send(F1i, dest=0, tag=501)
                    if has_f2: comm.Send(F2i, dest=0, tag=502)
                    if has_d: comm.Send(Di, dest=0, tag=503)
            if rank == 0:
                if r != 0:
                    Ur = np.empty((6, n, Ny, Nz), dtype=np.float32)
                    comm.Recv(Ur, source=r, tag=500)
                    F1r = np.empty((n, Ny, Nz), dtype=np.float32) if has_f1 else None
                    F2r = np.empty((n, Ny, Nz), dtype=np.float32) if has_f2 else None
                    Dr = np.empty((n, Ny, Nz, nbins), dtype=np.float32) if has_d else None
                    if has_f1: comm.Recv(F1r, source=r, tag=501)
                    if has_f2: comm.Recv(F2r, source=r, tag=502)
                    if has_d: comm.Recv(Dr, source=r, tag=503)

                if kind == "chk":
                    for iv, name in enumerate(("rho", "mx", "my", "mz", "E", "rho_eint")):
                        datasets[name][a:b, :, :] = Ur[iv]
                else:
                    for iv, name in enumerate(("U0", "U1", "U2", "U3", "U4", "U5")):
                        datasets[name][a:b, :, :] = Ur[iv]
                    rho, vx, vy, vz, P = eos.prim_from_cons(Ur)
                    eint = eos.internal_energy(Ur)
                    ekin = eos.kinetic_energy_density(Ur)
                    T = eos.temperature_from_pressure_density(P, rho)
                    vals = {
                        "dens": rho, "pres": P, "temp": T, "eint": eint, "ekin": ekin,
                        "etherm_frac": eint / np.maximum(np.abs(Ur[4]), 1.0e-30),
                        "ekin_over_eint": ekin / np.maximum(eint, 1.0e-30),
                        "velx": vx, "vely": vy, "velz": vz,
                    }
                    for name, arr in vals.items():
                        datasets[name][a:b, :, :] = np.asarray(arr, dtype=np.float32)
                if has_f1: f1ds[a:b, :, :] = F1r
                if has_f2: f2ds[a:b, :, :] = F2r
                if has_d:
                    for ib in range(nbins):
                        dds[ib][a:b, :, :] = Dr[..., ib]
        if rank == 0:
            h.close()
            os.replace(tmp, path)
            label = "[chk]" if kind == "chk" else ("[plt]" if kind == "plt" else "[hdf5]  ")
            logging.info(f"{label} wrote {path}")
    except Exception:
        if rank == 0:
            try: h.close()
            except Exception: pass
            try:
                if tmp and os.path.exists(tmp): os.remove(tmp)
            except Exception:
                pass
        raise
    finally:
        comm.Barrier()
    return path


def _write_snapshot_distributed(U, D_bins, f1, f2, nx_local, i0, i1, starts, counts,
                                comm, rank, t_sec, step, snap_id, plotting, initialize_plot_limits=False):
    path = _distributed_h5_path("std", snap_id, OUTDIR, BASEN)
    exists = comm.bcast(os.path.exists(path) if rank == 0 else None, root=0)
    if exists:
        comm.Barrier()
        return
    U2, D2, F1, F2 = _gather_view_slice(
        U, D_bins, f1, f2, nx_local, i0, i1, starts, counts, comm, rank
    )
    if rank == 0:
        plotting.make_png_from_slice(
            U2, VIEW_PLANE, t_sec / YR_S, snap_id, step, D2, F1, F2,
            initialize_plot_limits=initialize_plot_limits,
        )
    comm.Barrier()
    _write_h5_distributed(U, D_bins, f1, f2, nx_local, starts, counts,
                          comm, rank, t_sec, step, snap_id, kind="std")


def _write_checkpoint(U, D_bins, f1, f2, t, step, chk_id, snap_next, outdir, basen):
    """
    Self-contained checkpoint with plotfile-equivalent content + restart fields.
    Name: 3D_<basen>_hdf5_chk_####.
    Saves *both*:
      - 'chk'  : checkpoint id (#### above)
      - 'snap' : NEXT snapshot/plotfile id to use after restart
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"3D_{basen}_hdf5_chk_{int(chk_id):04d}")

    rho, vx, vy, vz, P = eos.prim_from_cons(U)
    ek = eos.kinetic_energy_density(U)
    eint = eos.internal_energy(U)
    T = eos.temperature_from_pressure_density(P, rho)

    with h5py.File(path, "w") as h:
        # core conserved fields
        h.create_dataset("rho", data=U[0], compression="gzip", compression_opts=1)
        h.create_dataset("mx",  data=U[1], compression="gzip", compression_opts=1)
        h.create_dataset("my",  data=U[2], compression="gzip", compression_opts=1)
        h.create_dataset("mz",  data=U[3], compression="gzip", compression_opts=1)
        h.create_dataset("E",   data=U[4], compression="gzip", compression_opts=1)
        h.create_dataset("rho_eint", data=(U[5] if U.shape[0] > 5 else eint), compression="gzip", compression_opts=1)

        # plotfile-like extras (so chk looks like plt):
        h.create_dataset("dens", data=U[0], compression="gzip", compression_opts=1)  # alias for rho
        h.create_dataset("vx",   data=vx,   compression="gzip", compression_opts=1)
        h.create_dataset("vy",   data=vy,   compression="gzip", compression_opts=1)
        h.create_dataset("vz",   data=vz,   compression="gzip", compression_opts=1)
        h.create_dataset("P",    data=P,    compression="gzip", compression_opts=1)
        h.create_dataset("T",    data=T,    compression="gzip", compression_opts=1)
        h.create_dataset("eint", data=eint, compression="gzip", compression_opts=1)
        h.create_dataset("ekin", data=ek, compression="gzip", compression_opts=1)
        h.create_dataset("etherm_frac", data=eint/np.maximum(np.abs(U[4]), 1e-30), compression="gzip", compression_opts=1)

        # tracers if present
        if f1 is not None: h.create_dataset("f1", data=f1, compression="gzip", compression_opts=1)
        if f2 is not None: h.create_dataset("f2", data=f2, compression="gzip", compression_opts=1)

        # dust bins if present
        NB = 0
        if D_bins is not None:
            NB = D_bins.shape[-1]
            for i in range(NB):
                h.create_dataset(f"dst{i}", data=D_bins[..., i], compression="gzip", compression_opts=1)
        h.attrs["NBINS"] = NB

        # run metadata (as datasets + a few attrs)
        h.create_dataset("time",  data=np.array(t, dtype=np.float64))
        h.create_dataset("nstep", data=np.array(step, dtype=np.int64))
        h.create_dataset("chk",   data=np.array(chk_id, dtype=np.int64))   # this checkpoint id
        h.create_dataset("snap",  data=np.array(snap_next, dtype=np.int64))# NEXT snapshot id
        h.create_dataset("frame_offset_cm", data=np.array(float(globals().get("FRAME_OFFSET_CM_RUNTIME", 0.0)), dtype=np.float64))
        h.create_dataset("frame_remainder_cm", data=np.array(float(globals().get("FRAME_REMAINDER_CM_RUNTIME", 0.0)), dtype=np.float64))
        h.attrs["frame_axis"] = str(globals().get("FRAME_AXIS_RUNTIME", globals().get("MOVING_FRAME_AXIS", "z")))

        # grid info for completeness
        h.attrs["dx"] = float(globals().get("dx", 0.0))
        h.attrs["dy"] = float(globals().get("dy", 0.0))
        h.attrs["dz"] = float(globals().get("dz", 0.0))
        h.attrs["Nx"] = int(globals().get("Nx", 0))
        h.attrs["Ny"] = int(globals().get("Ny", 0))
        h.attrs["Nz"] = int(globals().get("Nz", 0))
        h.attrs["(5.0/3.0)"] = float((5.0/3.0))

    return path

# ----------------------- CFL + stepping helpers -----------------------

def _cfl_dt(hydro, U):
    """
    Compute the CFL timestep using solver routines or a local fallback.
    """
    if hasattr(hydro, "cfl_dt"):
        return hydro.cfl_dt(U)
    if hasattr(hydro, "dt_cfl"):
        return hydro.dt_cfl(U)

    # Inline fallback
    rho, vx, vy, vz, P = eos.prim_from_cons(U)
    a   = np.sqrt(float(globals().get("(5.0/3.0)", 5.0/3.0))*np.maximum(P,1e-30)/rho)
    vmax_x = float(np.max(np.abs(vx) + a))
    vmax_y = float(np.max(np.abs(vy) + a))
    vmax_z = float(np.max(np.abs(vz) + a)) if DIM3D else 0.0
    dt_x = dx / vmax_x if vmax_x > 0.0 else 1e99
    dt_y = dy / vmax_y if vmax_y > 0.0 else 1e99
    dt_z = dz / vmax_z if DIM3D and vmax_z > 0.0 else 1e99
    return CFL * min(dt_x, dt_y, dt_z)

def _step_with_fluxes(hydro, U, dt):
    """
    Advance U by dt and try to obtain face mass fluxes for consistent scalar advection.
    Returns (U_new, (Fx,Fy,Fz)) when possible; otherwise (U_new, (None,None,None)).
    """
    U_prev = U
    # Request face fluxes from the hydrodynamic step.
    try:
        ret = hydro.step_euler(U, dt, return_fluxes=True)
        if isinstance(ret, tuple) and len(ret) == 2:
            return ret
        elif ret is not None:
            U = ret
        else:
            U = hydro.step_euler(U, dt)
    except TypeError:
        U = hydro.step_euler(U, dt)

    # If not provided, try helpers
    if hasattr(hydro, "_face_mass_fluxes"):
        try:
            Fx, Fy, Fz = hydro._face_mass_fluxes(U_prev)
            return U, (Fx, Fy, Fz)
        except Exception:
            pass
    if hasattr(hydro, "_face_fluxes"):
        try:
            Fx, Fy, Fz = hydro._face_fluxes(U_prev)
            return U, (Fx, Fy, Fz)
        except Exception:
            pass

    return U, (None, None, None)


def _publish_frame_state(frame_offset, frame_remainder, frame_axis):
    globals()["FRAME_OFFSET_CM_RUNTIME"] = float(frame_offset)
    globals()["FRAME_REMAINDER_CM_RUNTIME"] = float(frame_remainder)
    globals()["FRAME_AXIS_RUNTIME"] = str(frame_axis)
    try:
        from .plotting import plotting as _plotting
        _plotting.FRAME_OFFSET_CM_RUNTIME = float(frame_offset)
        _plotting.FRAME_REMAINDER_CM_RUNTIME = float(frame_remainder)
        _plotting.FRAME_AXIS_RUNTIME = str(frame_axis)
    except Exception:
        pass

# ----------------------- main -----------------------



def _build_get_D_local(D_bins):
    """Return a callable that sums dust tracer bins into total D field."""
    if D_bins is None:
        def _null():
            return 0.0
        return _null
    if isinstance(D_bins, (list, tuple)):
        def _fn():
            acc = None
            for b in D_bins:
                acc = b if acc is None else (acc + b)
            return acc
        return _fn
    else:
        def _fn():
            # assume last axis is bins
            import numpy as _np
            return _np.sum(D_bins, axis=-1)
        return _fn
def main(par_path="tonantzintla.par"):
    D_bins = None
    f1 = None
    f2 = None
    # Safe init to avoid UnboundLocalError before first use
    t = 0.0
    global kB, mH, mu_mol, MSUN, T_COLD_HOT_K

    # 1) Parse the parameter file independently on every rank.
    import hashlib
    import traceback

    try:
        cfg = read_data.load_config(par_path)
        settings_source = read_data.render_settings_module(cfg)
        local_report = (True, hashlib.sha256(settings_source.encode("utf-8")).hexdigest(), "")
    except Exception:
        cfg = None
        settings_source = None
        local_report = (False, "", traceback.format_exc())

    reports = COMM.gather(local_report, root=0)
    if RANK == 0:
        errors = [f"rank {rank}: {report[2]}" for rank, report in enumerate(reports) if not report[0]]
        digests = {report[1] for report in reports if report[0]}
        if errors:
            settings_status = (False, "Parameter loading failed:\n" + "\n".join(errors))
        elif len(digests) != 1:
            settings_status = (False, "MPI ranks produced different runtime settings.")
        else:
            settings_status = (True, next(iter(digests)))
    else:
        settings_status = None

    settings_ok, settings_info = COMM.bcast(settings_status, root=0)
    if not settings_ok:
        raise RuntimeError(settings_info)

    S = _install_settings_source(settings_source)
    if RANK == 0:
        try:
            read_data.write_settings_module(cfg, os.path.join(os.path.dirname(__file__), "settings.py"))
            write_status = (True, "")
        except Exception:
            write_status = (False, traceback.format_exc())
    else:
        write_status = None
    write_ok, write_error = COMM.bcast(write_status, root=0)
    if not write_ok:
        raise RuntimeError(f"Unable to write settings.py:\n{write_error}")

    # 2) Lazy imports (after settings exist) and lift constants into globals
    from .hydro import solvers as hydro
    from .clouds import clouds as clouds_mod
    from .cooling import cooling as cooling_mod
    from .plotting import plotting as plotting
    from .cinder import cinder
    from . import settings as settings
    globals().update({k: getattr(settings, k) for k in dir(settings) if not k.startswith('_')})
    # Normalize constants expected elsewhere
    kB    = globals().get('kB', 1.3806485e-16)
    mH    = globals().get('mH', 1.6726219e-24)
    mu_mol= globals().get('mu_mol', 2.33*mH)
    # Assert units: mu_mol is grams per particle (no extra mH factor)
    if not (np.isscalar(mu_mol) and (mu_mol > 0.0)):
        raise ValueError('mu_mol must be a positive scalar in grams per particle')
    if 'T_COLD_HOT_K' not in globals():
        Tc = globals().get('T_COLD_K', 300.0)
        Th = globals().get('T_HOT_K', 1.0e5)
        T_COLD_HOT_K = (Tc, Th)

    # Normalize constant names expected elsewhere
    globals()['(5.0/3.0)'] = globals().get('(5.0/3.0)', globals().get('(5.0/3.0)', 5.0/3.0))
    globals()['kB']    = globals().get('kB', 1.3806485e-16)
    globals()['mH']    = globals().get('mH', 1.6726219e-24)
    globals()['mu_mol']= globals().get('mu_mol', 2.33*globals()['mH'])
    # Temperature thresholds for phase split; fallback if not provided in settings
    if 'T_COLD_HOT_K' not in globals():
        Tc = globals().get('T_COLD_K', 300.0)
        Th = globals().get('T_HOT_K', 1.0e5)
        globals()['T_COLD_HOT_K'] = (Tc, Th)
    # --- MPI bootstrap ---
    i0, i1, nx_local, starts, counts = decompose_1d(Nx, SIZE, RANK)
    Ny_loc, Nz_loc = Ny, Nz
    def _alloc_local():  return np.zeros((6, nx_local+2, Ny_loc, Nz_loc), dtype=np.float32)
    def _alloc_scalar(): return np.zeros((nx_local+2, Ny_loc, Nz_loc), dtype=np.float32)
    

    os.makedirs(OUTDIR, exist_ok=True)

    # 3) Logging (to LOGFILE + console)
    _setup_logging_once(OUTDIR, BASEN, LOGFILE if 'LOGFILE' in globals() else None, (RESTART if 'RESTART' in locals() or 'RESTART' in globals() else False))
    log_header()

    # 4) Build dust bins (if enabled)
    bins = None
    if DUST_ON:
        bins = cinder.setup_bins()

    frame_axis = str(getattr(settings, "MOVING_FRAME_AXIS", "z")).lower()
    frame_v = float(getattr(settings, "MOVING_FRAME_V", 0.0))
    frame_offset = 0.0
    frame_remainder = 0.0
    _publish_frame_state(frame_offset, frame_remainder, frame_axis)

    # 5) Restart or distributed ICs (+ initial outputs)
    def _install_local_state(Ui, Di, F1i, F2i):
        Uloc = _alloc_local()
        Uloc[:, 1:1 + nx_local, :, :] = np.asarray(Ui, dtype=np.float32)

        if DUST_ON:
            nb = (Di.shape[-1] if Di is not None else (bins.NBINS if bins is not None else 0))
            Dloc = np.zeros((nx_local + 2, Ny, Nz, nb), dtype=np.float32) if nb else None
            if Di is not None:
                Dloc[1:1 + nx_local, :, :, :] = np.asarray(Di, dtype=np.float32)
        else:
            Dloc = None

        if TRACERS_ON:
            F1loc = _alloc_scalar()
            F2loc = _alloc_scalar()
            if F1i is not None:
                F1loc[1:1 + nx_local, :, :] = np.asarray(F1i, dtype=np.float32)
            if F2i is not None:
                F2loc[1:1 + nx_local, :, :] = np.asarray(F2i, dtype=np.float32)
        else:
            F1loc = F2loc = None

        exchange_ghosts_x(Uloc)
        if F1loc is not None: hydro.apply_tracer_bc(F1loc)
        if F2loc is not None: hydro.apply_tracer_bc(F2loc)
        if SIZE > 1 and Dloc is not None: exchange_ghosts_x_scalar(Dloc)
        return Uloc, Dloc, F1loc, F2loc

    fresh_start = not bool(RESTART)
    path = None
    if RESTART:
        if RANK == 0:
            snapreq = int(RESTART_SNAP) if isinstance(RESTART_SNAP, (int, float, str)) else 0
            if snapreq > 0:
                candidate = os.path.join(OUTDIR, f"3D_{BASEN}_hdf5_chk_{snapreq:04d}")
                if os.path.exists(candidate):
                    path = candidate
                else:
                    logging.warning(f"[restart] requested chk {snapreq:04d} not found: {candidate}. Falling back to latest.")
            if path is None:
                path = _find_latest_checkpoint(OUTDIR, BASEN)
        path = COMM.bcast(path, root=0)
        fresh_start = not (path and os.path.exists(path))

    if not fresh_start:
        if RANK == 0:
            logging.info(f"[restart] loading checkpoint slabs: {path}")
        (Ui, Di, F1i, F2i, t, step, loaded_chk, saved_snap_next,
         frame_offset, frame_remainder) = _load_checkpoint_local(path, i0, i1, COMM, RANK)
        U, D_bins, f1, f2 = _install_local_state(Ui, Di, F1i, F2i)
        del Ui, Di, F1i, F2i
        _publish_frame_state(frame_offset, frame_remainder, frame_axis)
        chk_id = int(loaded_chk) + 1
        if RANK == 0:
            snap_id_root = max(int(saved_snap_next), _next_standard_snap_id(OUTDIR, BASEN))
            if (SNAP_NEXT_OVERRIDE is not None) and (SNAP_NEXT_OVERRIDE >= 0):
                snap_id_root = int(SNAP_NEXT_OVERRIDE)
        else:
            snap_id_root = None
        snap_id = int(COMM.bcast(snap_id_root, root=0))
        if RANK == 0:
            logging.info(f"[restart] time={t/YR_S:.6e} yr  step={step}  next chk={chk_id:04d}  next snap={snap_id:04d}")
    else:
        if RESTART and RANK == 0:
            logging.info("[restart] no checkpoint found; starting from distributed ICs.")
        if RANK == 0:
            logging.info(f"[init] building distributed IC slabs on {SIZE} MPI ranks")
        Ui, Di, F1i, F2i, bins_ic = clouds_mod.init_state_local(
            i0, i1, comm=COMM, rank=RANK, bins=bins
        )
        if bins is None and bins_ic is not None:
            bins = bins_ic
        U, D_bins, f1, f2 = _install_local_state(Ui, Di, F1i, F2i)
        del Ui, Di, F1i, F2i
        t = 0.0
        step = 0
        snap_id = 0
        chk_id = 0
        if RANK == 0:
            logging.info("[init] local IC slabs ready; writing initial outputs")

        _write_snapshot_distributed(
            U, D_bins, f1, f2, nx_local, i0, i1, starts, counts,
            COMM, RANK, t, step, snap_id, plotting, initialize_plot_limits=True
        )
        snap_id += 1

        Mg_loc, Md_loc = clouds_mod.masses(
            U[:, 1:1 + nx_local, :, :],
            D_bins[1:1 + nx_local, :, :, :] if (DUST_ON and D_bins is not None) else None
        )
        Mg = COMM.reduce(Mg_loc, root=0)
        Md = COMM.reduce(Md_loc, root=0)
        U_sub = U[:, 1:1 + nx_local, :, :]
        D_sub = D_bins[1:1 + nx_local, :, :, :] if (DUST_ON and D_bins is not None) else None
        loc6 = compute_phase_masses_local(
            U_sub, D_sub, dx, dy, dz, (5.0 / 3.0), kB, mH, mu_mol, T_COLD_HOT_K
        )
        Gc, Gw, Gh, Dc, Dw, Dh = reduce6(COMM, loc6)
        allow_mass = _io_allow_this_step(COMM, RANK, t, YR_S, MASS_FILE)
        if allow_mass and RANK == 0:
            append_mass_row_once(MASS_FILE, 0.0, Mg, Md, Gc, Gw, Gh, Dc, Dw, Dh, MSUN)

        _write_h5_distributed(
            U, D_bins, f1, f2, nx_local, starts, counts,
            COMM, RANK, t, step, chk_id, kind="chk", snap_next=snap_id
        )
        chk_id += 1

    try:
        from .cooling import cooling as coolmod
        coolmod.get_D_local = _build_get_D_local(D_bins)
    except Exception:
        pass

    TEND = T_END_YR * YR_S

    # --- snapshot manager (time-based) ---
    try:
        snap_interval_sec = float(S.SNAP_EVERY) * float(YR_S)
    except Exception:
        snap_interval_sec = float(getattr(globals(), "SNAP_EVERY", 10.0)) * float(YR_S)
    SNAP = SnapshotManager(COMM, RANK, snap_interval_sec, phase_sec=t)

    # checkpoint cadence tracker
    last_chk_wall = time.monotonic()
    warned_no_flux = False

    # 6) Time loop
    # Defer periodic snapshots beyond the first time step when the cadence is shorter than that step.
    # The initial t=0 outputs remain unaffected.
    snap_gate_until_sec = None  # absolute time (sec) when periodic snaps may begin
    while t < TEND:
        dt = _cfl_dt(hydro, U); dt = min(dt, TEND - t)
        # Determine gate on the first iteration: if initial dt is longer than the requested cadence,
        # hold periodic snapshots until the simulation time reaches that initial dt.
        if (snap_gate_until_sec is None) and (step == 0):
            try:
                _snap_interval = float(S.SNAP_EVERY) * float(YR_S)
            except Exception:
                _snap_interval = float(getattr(globals(), 'SNAP_EVERY', 10.0)) * float(YR_S)
            if (_snap_interval > 0.0) and (dt > _snap_interval):
                snap_gate_until_sec = dt

        yr_s = (S.YR_S if 'S' in globals() else (YR_S if 'YR_S' in globals() else 365.0*24.0*3600.0))
        log_stride = int(globals().get('LOG_TIME_EVERY_STEPS', 1))
        if step % log_stride == 0:
            if RANK == 0: logging.info(f"[time] t = {t:.8e} s ({t/ yr_s:.6f} yr), step = {step}, dt = {dt:.8e} s")


        # --- keep rho^n (for consistent scalar transport) ---
        rho_old = hydro.apply_bc(U)[0].copy()

        # --- hydro step; try to get face fluxes for consistent advection ---
        U, (Fx, Fy, Fz) = _step_with_fluxes(hydro, U, dt)

        # --- tracers: consistent conservative advection with gas mass fluxes ---
        if TRACERS_ON:
            if Fx is not None:
                if f1 is not None:
                    f1 = hydro.advect_scalar_consistent(rho_old, U[0], f1, dt, Fx, Fy, Fz)
                if f2 is not None:
                    f2 = hydro.advect_scalar_consistent(rho_old, U[0], f2, dt, Fx, Fy, Fz)
            else:
                if not warned_no_flux:
                    logging.warning("No face fluxes available; skipping tracer advection this step.")
                    warned_no_flux = True

        # --- dust bins: consistent advection ---
        if DUST_ON and (bins is not None) and (D_bins is not None):
            if Fx is not None:
                D_bins = hydro.advect_scalar_consistent_multi(rho_old, U[0], D_bins, dt, Fx, Fy, Fz)
            else:
                if not warned_no_flux:
                    logging.warning("No face fluxes available; skipping dust advection this step.")
                    warned_no_flux = True
        # --- refresh scalar halos across X decomposition (if any) ---
        if TRACERS_ON:
            if f1 is not None: hydro.apply_tracer_bc(f1)
            if f2 is not None: hydro.apply_tracer_bc(f2)
        if SIZE > 1 and DUST_ON and (D_bins is not None):
            exchange_ghosts_x_scalar(D_bins)

# --- radiative + dust cooling ---
        if COOLING_ON:
            U = cooling_mod.apply_cooling(U, dt, bins, D_bins)        
        
        # --- dust growth + sputtering (updates D_bins mass in-place) ---
        if DUST_ON and (bins is not None) and (D_bins is not None):
            D_bins = cinder.apply_growth_and_sputtering(D_bins, U, dt, bins, mu_mol, kB, dx, dy, dz)
            if SIZE > 1:
                exchange_ghosts_x_scalar(D_bins)

        # advance time
        dt_last = dt  # capture step dt for snapshot timing
        t += dt
        step += 1
        if MOVING_FRAME_ON: frame_offset += frame_v*dt_last
        frame_remainder = 0.0
        _publish_frame_state(frame_offset, frame_remainder, frame_axis)

        # --- periodic mass & growth-zone stats ---
        if step % TSERIES_EVERY == 0:
            Mg_loc, Md_loc = clouds_mod.masses(U[:,1:1+nx_local,:,:], D_bins[1:1+nx_local,:,:,:] if (DUST_ON and (D_bins is not None)) else None)
            # phase masses (cold <5e3K, warm 5e3–1e4K, hot ≥1e4K)
            def _phase_masses_local(U_local, D_local):
                T1, T2 = 5.0e3, 1.0e4
                rho, vx, vy, vz, P = eos.prim_from_cons(U_local)
                T = eos.temperature_from_pressure_density(P, rho)
                vol   = float(dx)*float(dy)*float(dz)
                rho = np.asarray(rho, dtype=np.float64)
                m_cell = rho * vol
                m_g_c = np.sum(m_cell[T <  T1])
                m_g_w = np.sum(m_cell[(T >= T1) & (T < T2)])
                m_g_h = np.sum(m_cell[T >= T2])
                if D_local is not None:
                    m_d_cell = np.sum(D_local, axis=-1, dtype=np.float64) * m_cell
                    m_d_c = np.sum(m_d_cell[T <  T1])
                    m_d_w = np.sum(m_d_cell[(T >= T1) & (T < T2)])
                    m_d_h = np.sum(m_d_cell[T >= T2])
                else:
                    m_d_c = m_d_w = m_d_h = 0.0
                return (m_g_c, m_g_w, m_g_h, m_d_c, m_d_w, m_d_h)
            
            (g_c, g_w, g_h, d_c, d_w, d_h) = _phase_masses_local(U[:,1:1+nx_local,:,:], D_bins[1:1+nx_local,:,:,:] if (DUST_ON and (D_bins is not None)) else None)
            from .mpi_utils import COMM as _C
            Mg = _C.reduce(Mg_loc, root=0); Md = _C.reduce(Md_loc, root=0)
            Gc = _C.reduce(g_c, root=0);   Gw = _C.reduce(g_w, root=0);   Gh = _C.reduce(g_h, root=0)
            Dc = _C.reduce(d_c, root=0);   Dw = _C.reduce(d_w, root=0);   Dh = _C.reduce(d_h, root=0)
            _allow = _io_allow_this_step(COMM, RANK, t, YR_S, MASS_FILE)
            if _allow and RANK == 0:
                t_kyr = t / YR_S / 1e3
                append_mass_row_once(MASS_FILE, t_kyr, Mg, Md, Gc, Gw, Gh, Dc, Dw, Dh, MSUN)
            if DUST_ON and (D_bins is not None):
                Ui = U[:, 1:1+nx_local, :, :]
                Di = D_bins[1:1+nx_local, :, :, :]
                mask = cinder.growth_mask(Ui, mu_mol, kB)
                if np.any(mask):
                    Dtot = np.sum(np.clip(Di, 0, None), axis=-1)
                    num_loc = Di[mask].sum(axis=0)
                    den_loc = float(Dtot[mask].sum())
                else:
                    num_loc = np.zeros(Di.shape[-1], dtype=float)
                    den_loc = 0.0
                num = COMM.reduce(num_loc, root=0)
                den = COMM.reduce(den_loc, root=0)
                _allow = _io_allow_this_step(COMM, RANK, t, YR_S, MASS_FILE)
                if _allow and RANK == 0 and den > 0.0:
                    frac = num / max(den, 1e-30)
                    with open(GROWZONE_FILE, "a") as f:
                        f.write(f"{t/YR_S/1e3:12.5f}  " + "  ".join(f"{x: .6e}" for x in frac) + "\n")
        # --- periodic plots and snapshots ---
        snap_due = SNAP.due(dt_last)
        if snap_gate_until_sec is not None and t < snap_gate_until_sec:
            snap_due = False
        if snap_due:
            _write_snapshot_distributed(
                U, D_bins, f1, f2, nx_local, i0, i1, starts, counts,
                COMM, RANK, t, step, snap_id, plotting
            )
            snap_id += 1

        chk_every_steps = int(getattr(settings, "CHK_EVERY_STEPS", 0))
        chk_every_seconds = float(getattr(settings, "CHK_EVERY_SECONDS", 0.0))
        chk_due = COMM.bcast(((chk_every_steps > 0 and step > 0 and step % chk_every_steps == 0) or (chk_every_seconds > 0.0 and (time.monotonic() - last_chk_wall) >= chk_every_seconds)) if RANK == 0 else None, root=0)
        if chk_due:
            _write_h5_distributed(
                U, D_bins, f1, f2, nx_local, starts, counts,
                COMM, RANK, t, step, chk_id, kind="chk", snap_next=snap_id
            )
            chk_id += 1
            last_chk_wall = time.monotonic()

        flag_restart   = os.path.join(OUTDIR, ".dump_restart")
        flag_checkpoint= os.path.join(OUTDIR, ".dump_checkpoint")
        flag_plotfile  = os.path.join(OUTDIR, ".dump_plotfile")
        if RANK == 0:
            flags = (os.path.exists(flag_plotfile), os.path.exists(flag_checkpoint), os.path.exists(flag_restart))
        else:
            flags = None
        want_plotfile, want_checkpoint, want_restart = COMM.bcast(flags, root=0)

        if want_plotfile:
            _write_h5_distributed(
                U, D_bins, f1, f2, nx_local, starts, counts,
                COMM, RANK, t, step, snap_id, kind="plt"
            )
            if RANK == 0:
                _consume(flag_plotfile)
            COMM.Barrier()

        if want_checkpoint or want_restart:
            _write_h5_distributed(
                U, D_bins, f1, f2, nx_local, starts, counts,
                COMM, RANK, t, step, chk_id, kind="chk", snap_next=snap_id
            )
            chk_id += 1
            if RANK == 0:
                if want_checkpoint:
                    _consume(flag_checkpoint)
                if want_restart:
                    _consume(flag_restart)
            COMM.Barrier()
            if want_restart:
                if RANK == 0: logging.info("[restart] sentinel requested; checkpoint written. Exiting.")
                break

    logging.info("=== finished ===")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'tonantzintla.par')
