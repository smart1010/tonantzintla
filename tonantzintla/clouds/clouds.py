from ..cinder import cinder
from .. import eos
try:
    from ..settings import *
except Exception:
    pass
import logging
import numpy as np


def coordinates():
    """Return full-domain cell-centre coordinates (serial compatibility helper)."""
    x = (np.arange(Nx) + 0.5) * dx + XMIN_PC * PC_CM
    y = (np.arange(Ny) + 0.5) * dy + YMIN_PC * PC_CM
    z0 = (np.arange(Nz) + 0.5) * dz + ZMIN_PC * PC_CM if DIM3D else np.array([0.0])
    return np.meshgrid(x, y, z0, indexing="ij")


def coordinates_local(i0, i1):
    """Cell-centre coordinate vectors for the local X slab [i0:i1)."""
    x = (np.arange(int(i0), int(i1), dtype=np.float64) + 0.5) * dx + XMIN_PC * PC_CM
    y = (np.arange(Ny, dtype=np.float64) + 0.5) * dy + YMIN_PC * PC_CM
    z = ((np.arange(Nz, dtype=np.float64) + 0.5) * dz + ZMIN_PC * PC_CM
         if DIM3D else np.array([0.0], dtype=np.float64))
    return x, y, z


# --------------------------- TURBULENCE (Kolmogorov) -----------------------

def _kolmogorov_raw_local(x, y, z, seed, nmodes=32):
    """Generate the unnormalised deterministic velocity modes on one X slab."""
    shape = (x.size, y.size, z.size)
    rng = np.random.default_rng(seed)
    Lx = (XMAX_PC - XMIN_PC) * PC_CM
    Ly = (YMAX_PC - YMIN_PC) * PC_CM
    Lz = (ZMAX_PC - ZMIN_PC) * PC_CM
    kmin = 2.0 * np.pi / min(Lx, Ly, Lz)
    kmax = 2.0 * np.pi / max(dx, dy, dz)

    vx = np.zeros(shape, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    vz = np.zeros(shape, dtype=np.float32)

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    for _ in range(int(nmodes)):
        u = rng.random()
        k = np.exp(np.log10(kmin) + u * (np.log10(kmax) - np.log10(kmin)))
        th = np.arccos(2.0 * rng.random() - 1.0)
        ph = 2.0 * np.pi * rng.random()
        kx = k * np.sin(th) * np.cos(ph)
        ky = k * np.sin(th) * np.sin(ph)
        kz = k * np.cos(th)

        a_th = 2.0 * np.pi * rng.random()
        kk = np.array([kx, ky, kz], dtype=np.float64)
        e = np.array([np.cos(a_th), np.sin(a_th), 0.0], dtype=np.float64)
        e -= (np.dot(e, kk) / max(np.dot(kk, kk), 1.0e-30)) * kk
        e /= np.sqrt(max(np.dot(e, e), 1.0e-30))

        amp = k ** (-11.0 / 6.0)
        phase = 2.0 * np.pi * rng.random()
        arg = kx * X + ky * Y + kz * Z + phase
        s = np.sin(arg)
        vx += (amp * e[0] * s).astype(np.float32, copy=False)
        vy += (amp * e[1] * s).astype(np.float32, copy=False)
        vz += (amp * e[2] * s).astype(np.float32, copy=False)

    return vx, vy, vz


def kolmogorov_velocity_field_local(x, y, z, sigma_kms, seed, comm=None, nmodes=32):
    """Velocity modes with global RMS normalisation across slabs."""
    shape = (x.size, y.size, z.size)
    if sigma_kms <= 0.0:
        return (np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32))

    vx, vy, vz = _kolmogorov_raw_local(x, y, z, seed, nmodes=nmodes)
    local_ss = float(np.sum(vx.astype(np.float64) ** 2 +
                            vy.astype(np.float64) ** 2 +
                            vz.astype(np.float64) ** 2))
    local_n = int(vx.size)
    if comm is not None:
        global_ss = float(comm.allreduce(local_ss))
        global_n = int(comm.allreduce(local_n))
    else:
        global_ss = local_ss
        global_n = local_n

    rms = np.sqrt(global_ss / max(global_n, 1))
    if rms > 0.0:
        scale = np.float32((sigma_kms * 1.0e5) / rms)
        vx *= scale
        vy *= scale
        vz *= scale
    return vx, vy, vz


def kolmogorov_velocity_field(X, Y, Z, sigma_kms, seed, nmodes=32):
    """Serial compatibility wrapper for callers using full coordinate meshes."""
    if sigma_kms <= 0.0:
        return np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    x = np.asarray(X[:, 0, 0], dtype=np.float64)
    y = np.asarray(Y[0, :, 0], dtype=np.float64)
    z = np.asarray(Z[0, 0, :], dtype=np.float64)
    return kolmogorov_velocity_field_local(x, y, z, sigma_kms, seed, comm=None, nmodes=nmodes)


def add_turbulence_by_region_local(vx, vy, vz, x, y, z, mask, sigma_kms, seed, comm=None):
    if sigma_kms <= 0.0:
        return vx, vy, vz
    vx_t, vy_t, vz_t = kolmogorov_velocity_field_local(
        x, y, z, sigma_kms, seed, comm=comm
    )
    vx[mask] += vx_t[mask]
    vy[mask] += vy_t[mask]
    vz[mask] += vz_t[mask]
    return vx, vy, vz


def add_turbulence_by_region(vx, vy, vz, X, Y, Z, mask, sigma_kms, seed):
    if sigma_kms <= 0.0:
        return vx, vy, vz
    vx_t, vy_t, vz_t = kolmogorov_velocity_field(X, Y, Z, sigma_kms, seed)
    vx[mask] += vx_t[mask]
    vy[mask] += vy_t[mask]
    vz[mask] += vz_t[mask]
    return vx, vy, vz


# --------------------------- INITIAL CONDITIONS -----------------------------

def _blend_inplace(field, target, alpha):
    """Blend a positive scalar with a common interface weight."""
    field *= (1.0 - alpha)
    field += np.float32(target) * alpha


def _ramp_cells(rho_outer, rho_inner, radius):
    mode = str(globals().get("INTERFACE_RAMP_MODE", "auto")).lower()
    if radius <= 0.0 or mode == "sharp":
        return 0
    if mode == "fixed":
        ncell = int(globals().get("INTERFACE_RAMP_CELLS", 2))
    else:
        rho0 = max(float(rho_outer), np.finfo(np.float32).tiny)
        rho1 = max(float(rho_inner), np.finfo(np.float32).tiny)
        dex = abs(np.log10(rho1 / rho0))
        max_dex = float(globals().get("INTERFACE_RAMP_MAX_DEX_PER_CELL", 1.0))
        ncell = int(np.ceil(0.5 * np.pi * dex / max(max_dex, 1.0e-12)))
        ncell = max(ncell, int(globals().get("INTERFACE_RAMP_MIN_CELLS", 2)))
        ncell = min(ncell, int(globals().get("INTERFACE_RAMP_MAX_CELLS", 16)))
    geometric_max = max(1, int(np.floor(2.0 * radius / max(dx, dy, dz) - 1.0e-12)))
    return max(1, min(ncell, geometric_max))


def _interface_alpha(distance, radius, ncell):
    if ncell <= 0:
        return (distance <= radius).astype(np.float32)
    width = max(float(ncell) * max(dx, dy, dz), 1.0e-30)
    s = np.clip(0.5 - (distance - radius) / width, 0.0, 1.0)
    return (0.5 * (1.0 - np.cos(np.pi * s))).astype(np.float32, copy=False)


def _blend_cloud_state(rho, vx, vy, vz, pressure, alpha,
                       target_rho, target_vx, target_vy, target_vz, target_pressure):
    """Blend log density and sqrt(rho)*velocity without kinetic-energy overshoot."""
    one_minus = np.float32(1.0) - alpha
    rho_floor = np.finfo(np.float32).tiny
    rho_old = np.maximum(rho, rho_floor)
    sqrt_old = np.sqrt(rho_old)
    log_rho = np.log(rho_old) * one_minus
    log_rho += alpha * np.float32(np.log(max(float(target_rho), rho_floor)))
    rho_new = np.exp(log_rho).astype(np.float32, copy=False)
    sqrt_new = np.sqrt(np.maximum(rho_new, rho_floor))
    sqrt_target = np.float32(np.sqrt(max(float(target_rho), rho_floor)))

    for velocity, target_velocity in ((vx, target_vx), (vy, target_vy), (vz, target_vz)):
        velocity *= sqrt_old
        velocity *= one_minus
        velocity += alpha * sqrt_target * np.float32(target_velocity)
        velocity /= sqrt_new

    rho[...] = rho_new
    _blend_inplace(pressure, target_pressure, alpha)


def init_state_local(i0, i1, comm=None, rank=0, bins=None):
    """Construct the local X slab [i0:i1) without ghost cells."""
    i0 = int(i0)
    i1 = int(i1)
    nx = i1 - i0
    x, y, z = coordinates_local(i0, i1)
    shape = (nx, Ny, Nz)

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    shift_x = MOVING_FRAME_V if MOVING_FRAME_ON and MOVING_FRAME_AXIS == "x" else 0.0
    shift_y = MOVING_FRAME_V if MOVING_FRAME_ON and MOVING_FRAME_AXIS == "y" else 0.0
    shift_z = MOVING_FRAME_V if MOVING_FRAME_ON and MOVING_FRAME_AXIS == "z" else 0.0
    if MOVING_FRAME_ON and MOVING_FRAME_AXIS not in {"x", "y", "z"}:
        raise ValueError("MOVING_FRAME_AXIS must be 'x', 'y', or 'z'")

    vx_amb = VX_AMB_KMS * 1.0e5 - shift_x
    vy_amb = VY_AMB_KMS * 1.0e5 - shift_y
    vz_amb = VZ_AMB_KMS * 1.0e5 - shift_z
    rho = np.full(shape, RHO_AMB, dtype=np.float32)
    vx = np.full(shape, vx_amb, dtype=np.float32)
    vy = np.full(shape, vy_amb, dtype=np.float32)
    vz = np.full(shape, vz_amb, dtype=np.float32)
    P = np.full(shape, P_AMB, dtype=np.float32)

    r1 = C1_RADIUS_PC * PC_CM
    if r1 > 0.0:
        c1 = (C1_CENTER_PC[0] * PC_CM,
              C1_CENTER_PC[1] * PC_CM,
              C1_CENTER_PC[2] * PC_CM)
        dist1_sq = (X - c1[0]) ** 2 + (Y - c1[1]) ** 2 + (Z - c1[2]) ** 2
        dist1 = np.sqrt(dist1_sq)
        mask1 = dist1_sq <= r1 * r1
        ramp1 = _ramp_cells(RHO_AMB, C1_RHO, r1)
        alpha1 = _interface_alpha(dist1, r1, ramp1)
        _blend_cloud_state(
            rho, vx, vy, vz, P, alpha1,
            C1_RHO, C1_VX - shift_x, C1_VY - shift_y, C1_VZ - shift_z,
            eos.pressure_from_density_temperature(C1_RHO, C1_T),
        )
        del dist1_sq, dist1
    else:
        ramp1 = 0
        mask1 = np.zeros(shape, dtype=bool)
        alpha1 = np.zeros(shape, dtype=np.float32)

    r2 = C2_RADIUS_PC * PC_CM
    if r2 > 0.0:
        c2 = (C2_CENTER_PC[0] * PC_CM,
              C2_CENTER_PC[1] * PC_CM,
              C2_CENTER_PC[2] * PC_CM)
        dist2_sq = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2 + (Z - c2[2]) ** 2
        dist2 = np.sqrt(dist2_sq)
        mask2 = dist2_sq <= r2 * r2
        ramp2 = _ramp_cells(RHO_AMB, C2_RHO, r2)
        alpha2 = _interface_alpha(dist2, r2, ramp2)
        _blend_cloud_state(
            rho, vx, vy, vz, P, alpha2,
            C2_RHO, C2_VX - shift_x, C2_VY - shift_y, C2_VZ - shift_z,
            eos.pressure_from_density_temperature(C2_RHO, C2_T),
        )
        del dist2_sq, dist2
    else:
        ramp2 = 0
        mask2 = np.zeros(shape, dtype=bool)
        alpha2 = np.zeros(shape, dtype=np.float32)

    if int(rank) == 0:
        logging.info(f"[init] cloud1 interface ramp: {ramp1} cells")
        if C2_RADIUS_PC > 0.0:
            logging.info(f"[init] cloud2 interface ramp: {ramp2} cells")

    if TURB_ON:
        mask_amb = ~(mask1 | mask2)
        vx, vy, vz = add_turbulence_by_region_local(
            vx, vy, vz, x, y, z, mask_amb,
            SIGMA_T_AMB_KMS, TURB_SEED + 1, comm=comm
        )
        vx, vy, vz = add_turbulence_by_region_local(
            vx, vy, vz, x, y, z, mask1,
            SIGMA_T_C1_KMS, TURB_SEED + 2, comm=comm
        )
        vx, vy, vz = add_turbulence_by_region_local(
            vx, vy, vz, x, y, z, mask2,
            SIGMA_T_C2_KMS, TURB_SEED + 3, comm=comm
        )

    fcl1 = alpha1.copy()
    fcl2 = alpha2.copy()

    if DUST_ON:
        if bins is None:
            bins = cinder.setup_bins()
        if int(rank) == 0:
            cinder.print_bins_table(bins)
        D_bins = np.empty(shape + (bins.NBINS,), dtype=np.float32)
        for ib in range(bins.NBINS):
            frac = np.float32(bins.mass_fracs[ib])
            Di = np.full(shape, np.float32(D_INIT_AMBIENT) * frac, dtype=np.float32)
            _blend_inplace(Di, np.float32(D_INIT_CLOUD1) * frac, alpha1)
            _blend_inplace(Di, np.float32(D_INIT_CLOUD2) * frac, alpha2)
            D_bins[..., ib] = Di
        if int(rank) == 0:
            with open(GROWZONE_FILE, "w") as f:
                f.write("# t[kyr]  " + "  ".join([f"frac{i}" for i in range(bins.NBINS)]) + "\n")
    else:
        D_bins = None
        bins = None

    if int(rank) == 0:
        with open(MASS_FILE, "w") as f:
            f.write("# t[kyr]   M_gas[Msun]   M_dust[Msun]\n")

    eint = (P / np.float32(gamma - 1.0)).astype(np.float32, copy=False)
    E = (eint + np.float32(0.5) * rho * (vx * vx + vy * vy + vz * vz)).astype(np.float32, copy=False)
    U = np.empty((6,) + shape, dtype=np.float32)
    U[0] = rho
    U[1] = rho * vx
    U[2] = rho * vy
    U[3] = rho * vz
    U[4] = E
    U[5] = eint
    return U, D_bins, fcl1, fcl2, bins


def init_state():
    """Serial full-domain compatibility wrapper."""
    return init_state_local(0, Nx, comm=None, rank=0, bins=None)


# ------------------------- PRIMITIVES / FLUXES ------------------------------

def masses(U, D_bins):
    rho = np.asarray(U[0], dtype=np.float64)
    M_gas = rho.sum() * float(VOL_CELL) / float(MSUN)
    if D_bins is None:
        M_dust = 0.0
    else:
        Dtot = np.sum(np.clip(D_bins, 0, None), axis=-1, dtype=np.float64)
        M_dust = (rho * Dtot).sum() * float(VOL_CELL) / float(MSUN)
    return M_gas, M_dust
