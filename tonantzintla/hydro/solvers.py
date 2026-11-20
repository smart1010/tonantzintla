try:
    from ..settings import *
except ModuleNotFoundError:
    # settings is generated later by read_data; defer usage until runtime
    pass
from ..mpi_utils import (
    COMM, RANK, SIZE, apply_bc_mpi, exchange_ghosts_x, exchange_ghosts_x_scalar,
    exchange_right_face_state_x, synchronize_left_owned_flux_x,
)
import numpy as np
from .. import eos
# Simple buffer cache to reduce repeated allocations
_BUF = {}

def _buf_like(key, ref):
    shp = ref.shape; dt = ref.dtype
    k = (key, shp, dt)
    arr = _BUF.get(k)
    if arr is None:
        arr = np.zeros(shp, dtype=dt)
        _BUF[k] = arr
    return arr


# ==== slope-limiter helpers (global scope) ====
def _minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def _mc(a, b):
    mm = _minmod(a, b)
    return _minmod(0.5*(a + b), 2.0*mm)
# =============================================
try:
    from ..cinder import cinder
except Exception:
    cinder = None

# per-timestep scalar-call counter for wind bin indexing (reset in step_euler)
_WIND_SCALAR_CALL_COUNT = 0


# ------------------ Wind tunnel helpers ------------------
_mu_piecewise = eos.mu_from_temperature


def _wind_inflow_state():
    """Build conserved U for the wind ghost cells from S."""
    from .. import settings as S
    if not getattr(S, 'WIND_ON', False):
        return None
    # velocities (prefer cgs if present, else derive from *_KMS)
    vx = getattr(S, 'WIND_VX', None)
    vy = getattr(S, 'WIND_VY', None)
    vz = getattr(S, 'WIND_VZ', None)
    if vx is None: vx = getattr(S, 'WIND_VX_KMS', 0.0) * 1.0e5 if vx is None else vx
    if vy is None: vy = getattr(S, 'WIND_VY_KMS', 0.0) * 1.0e5 if vy is None else vy
    if vz is None: vz = getattr(S, 'WIND_VZ_KMS', 0.0) * 1.0e5 if vz is None else vz

    T  = getattr(S, 'WIND_T', 1.0e7)
    n  = getattr(S, 'WIND_N_CM3', 1.0e-2)  # total number density [cm^-3]
    mu = _mu_piecewise(T)
    rho = np.maximum(mu * n, 1e-30)
    P   = np.maximum(n * S.kB * T, 1e-30)
    face = getattr(S, 'WIND_FACE', 'x-')
    if face == 'x-':   vx = abs(vx)
    elif face == 'x+': vx = -abs(vx)
    elif face == 'y-': vy = abs(vy)
    elif face == 'y+': vy = -abs(vy)
    elif face == 'z-': vz = abs(vz)
    elif face == 'z+': vz = -abs(vz)

    if getattr(S, 'MOVING_FRAME_ON', False):
        if S.MOVING_FRAME_AXIS == 'x': vx -= S.MOVING_FRAME_V
        elif S.MOVING_FRAME_AXIS == 'y': vy -= S.MOVING_FRAME_V
        elif S.MOVING_FRAME_AXIS == 'z': vz -= S.MOVING_FRAME_V
        else: raise ValueError("MOVING_FRAME_AXIS must be 'x', 'y', or 'z'")

    E = P/(getattr(S, 'gamma', 5.0/3.0) - 1.0) + 0.5*rho*(vx*vx + vy*vy + vz*vz)
    U_in = np.zeros((6,), dtype=float)
    U_in[0] = rho
    U_in[1] = rho*vx
    U_in[2] = rho*vy
    U_in[3] = rho*vz
    U_in[4] = E
    U_in[5] = P/(getattr(S,'gamma',5.0/3.0) - 1.0)
    return U_in


def _ambient_inflow_state():
    from .. import settings as S
    rho = max(float(getattr(S, 'RHO_AMB', 1.0e-30)), 1.0e-30)
    vx = float(getattr(S, 'VX_AMB_KMS', 0.0))*1.0e5
    vy = float(getattr(S, 'VY_AMB_KMS', 0.0))*1.0e5
    vz = float(getattr(S, 'VZ_AMB_KMS', 0.0))*1.0e5
    if getattr(S, 'MOVING_FRAME_ON', False):
        if S.MOVING_FRAME_AXIS == 'x': vx -= S.MOVING_FRAME_V
        elif S.MOVING_FRAME_AXIS == 'y': vy -= S.MOVING_FRAME_V
        elif S.MOVING_FRAME_AXIS == 'z': vz -= S.MOVING_FRAME_V
        else: raise ValueError("MOVING_FRAME_AXIS must be 'x', 'y', or 'z'")
    P = max(float(getattr(S, 'P_AMB', eos.pressure_from_density_temperature(rho, getattr(S, 'T_AMB', 1.0e4)))), 1.0e-30)
    eint = P/(getattr(S, 'gamma', 5.0/3.0) - 1.0)
    return np.array([rho, rho*vx, rho*vy, rho*vz, eint + 0.5*rho*(vx*vx + vy*vy + vz*vz), eint], dtype=float)


def _boundary_inflow_state():
    from .. import settings as S
    if getattr(S, 'WIND_ON', False):
        return getattr(S, 'WIND_FACE', 'x-'), _wind_inflow_state()
    if not getattr(S, 'MOVING_FRAME_ON', False):
        return None, None
    U_in = _ambient_inflow_state()
    axis = str(S.MOVING_FRAME_AXIS).lower()
    vn = U_in[{'x': 1, 'y': 2, 'z': 3}[axis]]/U_in[0]
    return (axis + ('-' if vn > 0.0 else '+'), U_in) if vn != 0.0 else (None, None)


# --------------------------- PRIMITIVES / FLUXES ----------------------------
def prim_from_cons(U):
    from .. import settings
    """Return (rho, vx, vy, vz, P) from conserved U=[rho,mx,my,mz,E]."""
    rho = np.maximum(U[0], 1e-30)
    vx  = U[1]/rho
    vy  = U[2]/rho
    vz  = U[3]/rho
    ek  = 0.5*rho*(vx*vx+vy*vy+vz*vz)
    P   = eos.pressure_from_cons(U)
    return rho, vx, vy, vz, P


def cons_from_prim(rho, vx, vy, vz, P, nvar=6):
    from .. import settings
    U = np.empty((int(nvar),)+rho.shape, dtype=np.result_type(rho, vx, vy, vz, P))
    U[0] = rho
    U[1] = rho*vx
    U[2] = rho*vy
    U[3] = rho*vz
    U[4] = P/(settings.gamma - 1.0) + 0.5*rho*(vx*vx+vy*vy+vz*vz)
    if U.shape[0] > 5:
        U[5] = P/(settings.gamma - 1.0)
    return U


def sound_speed(P, rho):
    from .. import settings
    """a = sqrt(gamma*P/rho), with argument order (P, rho)."""
    return np.sqrt(settings.gamma*np.maximum(P,1e-30)/np.maximum(rho,1e-30))




def _limited_slope(q, axis, floor=None):
    s = np.zeros_like(q)
    if axis == 'x':
        s[1:-1] = _mc(q[1:-1] - q[:-2], q[2:] - q[1:-1])
    elif axis == 'y':
        s[:, 1:-1] = _mc(q[:, 1:-1] - q[:, :-2], q[:, 2:] - q[:, 1:-1])
    elif axis == 'z':
        s[:, :, 1:-1] = _mc(q[:, :, 1:-1] - q[:, :, :-2], q[:, :, 2:] - q[:, :, 1:-1])
    else:
        raise ValueError("axis must be 'x','y' or 'z'")
    if floor is not None:
        theta = np.minimum(1.0, 2.0*np.maximum(q - floor, 0.0)/np.maximum(np.abs(s), 1.0e-30))
        s *= theta
    return s


def _primitive_slopes(U):
    q = prim_from_cons(U)
    slopes = {}
    for axis in ('x', 'y', 'z'):
        slopes[axis] = tuple(
            _limited_slope(field, axis, 1.0e-30 if i in (0, 4) else None)
            for i, field in enumerate(q)
        )
    return q, slopes


def _edge_cons(q, slopes, axis, sign, nvar):
    qe = [field + 0.5*sign*slope for field, slope in zip(q, slopes[axis])]
    qe[0] = np.maximum(qe[0], 1.0e-30)
    qe[4] = np.maximum(qe[4], 1.0e-30)
    return cons_from_prim(*qe, nvar=nvar)


def _hancock_predictor(U, dt, q, slopes):
    """Predict cell averages to t + dt/2 from limited edge states."""
    from .. import settings as S
    rhs = np.zeros_like(U)
    for axis, spacing in (('x', S.dx), ('y', S.dy)):
        Um = _edge_cons(q, slopes, axis, -1.0, U.shape[0])
        Up = _edge_cons(q, slopes, axis, +1.0, U.shape[0])
        rhs -= (_flux_components(Up, axis) - _flux_components(Um, axis))/spacing
    if bool(getattr(S, 'DIM3D', False)):
        Um = _edge_cons(q, slopes, 'z', -1.0, U.shape[0])
        Up = _edge_cons(q, slopes, 'z', +1.0, U.shape[0])
        rhs -= (_flux_components(Up, 'z') - _flux_components(Um, 'z'))/S.dz
    U_half = U + 0.5*dt*rhs
    if U.shape[0] > 5:
        divv = slopes['x'][1]/S.dx + slopes['y'][2]/S.dy
        if bool(getattr(S, 'DIM3D', False)): divv += slopes['z'][3]/S.dz
        U_half[5] -= 0.5*dt*q[4]*divv
    return eos.reconcile_dual_energy(U_half)


def _slice_faces(q, s, axis):
    if axis == 'x': return q[:-1] + 0.5*s[:-1], q[1:] - 0.5*s[1:]
    if axis == 'y': return q[:, :-1] + 0.5*s[:, :-1], q[:, 1:] - 0.5*s[:, 1:]
    if axis == 'z': return q[:, :, :-1] + 0.5*s[:, :, :-1], q[:, :, 1:] - 0.5*s[:, :, 1:]
    raise ValueError("axis must be 'x','y' or 'z'")


def _reconstruct_prims_plm(U_half, slopes, axis):
    """MUSCL-Hancock face states with a local first-order positivity fallback."""
    from .. import settings as S
    q_half = prim_from_cons(U_half)
    face_slopes = [s.copy() for s in slopes[axis]]
    frac = float(getattr(S, 'HANCOCK_FLATTEN_FRACTION', 0.1))
    rho_min = np.maximum(frac*q_half[0], 1.0e-30)
    p_min = np.maximum(frac*q_half[4], 1.0e-30)
    bad = ((q_half[0] - 0.5*np.abs(face_slopes[0]) <= rho_min) |
           (q_half[4] - 0.5*np.abs(face_slopes[4]) <= p_min))
    for field, slope in zip(q_half, face_slopes):
        bad |= ~np.isfinite(field) | ~np.isfinite(slope)
    for slope in face_slopes:
        slope[bad] = 0.0
    faces = [_slice_faces(field, slope, axis) for field, slope in zip(q_half, face_slopes)]
    left = [pair[0] for pair in faces]
    right = [pair[1] for pair in faces]
    return cons_from_prim(*left, nvar=U_half.shape[0]), cons_from_prim(*right, nvar=U_half.shape[0])


def _reconstruct_prims_first(U, axis):
    """Piecewise-constant (first-order) reconstruction (Godunov)."""
    rho, vx, vy, vz, P = prim_from_cons(U)
    if axis == 'x':
        rhoL, rhoR = rho[0:-1,:,:], rho[1:  ,:,:]
        vxL,  vxR  = vx [0:-1,:,:], vx [1:  ,:,:]
        vyL,  vyR  = vy [0:-1,:,:], vy [1:  ,:,:]
        vzL,  vzR  = vz [0:-1,:,:], vz [1:  ,:,:]
        PL,   PR   = P  [0:-1,:,:], P  [1:  ,:,:]
    elif axis == 'y':
        rhoL, rhoR = rho[:,0:-1,:], rho[:,1:  ,:]
        vxL,  vxR  = vx [:,0:-1,:], vx [:,1:  ,:]
        vyL,  vyR  = vy [:,0:-1,:], vy [:,1:  ,:]
        vzL,  vzR  = vz [:,0:-1,:], vz [:,1:  ,:]
        PL,   PR   = P  [:,0:-1,:], P  [:,1:  ,:]
    elif axis == 'z':
        rhoL, rhoR = rho[:,:,0:-1], rho[:,:,1:  ]
        vxL,  vxR  = vx [:,:,0:-1], vx [:,:,1:  ]
        vyL,  vyR  = vy [:,:,0:-1], vy [:,:,1:  ]
        vzL,  vzR  = vz [:,:,0:-1], vz [:,:,1:  ]
        PL,   PR   = P  [:,:,0:-1], P  [:,:,1:  ]
    else:
        raise ValueError("axis must be 'x','y' or 'z'")

    UL = cons_from_prim(rhoL, vxL, vyL, vzL, PL, U.shape[0])
    UR = cons_from_prim(rhoR, vxR, vyR, vzR, PR, U.shape[0])
    if U.shape[0] > 5:
        if axis == 'x': UL[5:], UR[5:] = U[5:,0:-1,:,:], U[5:,1:,:,:]
        elif axis == 'y': UL[5:], UR[5:] = U[5:,:,0:-1,:], U[5:,:,1:,:]
        else: UL[5:], UR[5:] = U[5:,:,:,0:-1], U[5:,:,:,1:]
    return UL, UR


def _flux_components(U, axis):
    """Return physical flux F(U) along the selected axis."""
    rho, vx, vy, vz, P = prim_from_cons(U)
    E = U[4]
    if axis == 'x':
        vn = vx
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx + P
        F[2] = rho*vn*vy
        F[3] = rho*vn*vz
        F[4] = (E + P)*vn
        if U.shape[0] > 5: F[5:] = U[5:]*vn
        return F
    if axis == 'y':
        vn = vy
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx
        F[2] = rho*vn*vy + P
        F[3] = rho*vn*vz
        F[4] = (E + P)*vn
        if U.shape[0] > 5: F[5:] = U[5:]*vn
        return F
    if axis == 'z':
        vn = vz
        F = np.empty_like(U)
        F[0] = rho*vn
        F[1] = rho*vn*vx
        F[2] = rho*vn*vy
        F[3] = rho*vn*vz + P
        F[4] = (E + P)*vn
        if U.shape[0] > 5: F[5:] = U[5:]*vn
        return F
    raise ValueError("axis must be 'x','y' or 'z'")


def _hlle(UL, UR, axis):
    """
    HLLE flux between left/right conserved states along given axis.
    Returns the flux vector and its mass-flux component.
    """
    # primitives
    rhoL, vxL, vyL, vzL, PL = prim_from_cons(UL)
    rhoR, vxR, vyR, vzR, PR = prim_from_cons(UR)

    if axis == 'x':
        vnL, vnR = vxL, vxR
    elif axis == 'y':
        vnL, vnR = vyL, vyR
    else:
        vnL, vnR = vzL, vzR

    aL = sound_speed(PL, rhoL)
    aR = sound_speed(PR, rhoR)

    sL = np.minimum(vnL - aL, vnR - aR)
    sR = np.maximum(vnL + aL, vnR + aR)

    FL = _flux_components(UL, axis)
    FR = _flux_components(UR, axis)

    # upwind cases
    F = np.where(sL >= 0.0, FL, np.where(sR <= 0.0, FR,
         (sR*FL - sL*FR + (sL*sR)*(UR - UL)) / np.maximum(sR - sL, 1e-30)))
    mass_flux = F[0]
    return F, mass_flux


# --------------------------- RIEMANN / FACE FLUXES --------------------------

def _safe_denom(a, floor=1.0e-30):
    return np.where(np.abs(a) > floor, a, np.where(a >= 0.0, floor, -floor))


def _star_state(U, P, S, SM, rho_star, axis):
    rho, vx, vy, vz, _ = prim_from_cons(U)
    vn = vx if axis == 'x' else (vy if axis == 'y' else vz)
    p_star = P + rho*(S - vn)*(SM - vn)
    U_star = np.empty_like(U)
    U_star[0] = rho_star
    U_star[1] = rho_star*(SM if axis == 'x' else vx)
    U_star[2] = rho_star*(SM if axis == 'y' else vy)
    U_star[3] = rho_star*(SM if axis == 'z' else vz)
    U_star[4] = ((S - vn)*U[4] - P*vn + p_star*SM)/_safe_denom(S - SM)
    if U.shape[0] > 5:
        U_star[5:] = rho_star[None, ...]*(U[5:]/np.maximum(rho, 1.0e-30)[None, ...])
    return U_star, p_star


def _hllc(UL, UR, axis):
    """HLLC flux with face-local HLLE fallback for nonphysical star states."""
    face_shape = np.broadcast_shapes(UL.shape[1:], UR.shape[1:])
    UL = np.broadcast_to(UL, (UL.shape[0],) + face_shape)
    UR = np.broadcast_to(UR, (UR.shape[0],) + face_shape)
    rhoL, vxL, vyL, vzL, PL = prim_from_cons(UL)
    rhoR, vxR, vyR, vzR, PR = prim_from_cons(UR)
    vnL = vxL if axis == 'x' else (vyL if axis == 'y' else vzL)
    vnR = vxR if axis == 'x' else (vyR if axis == 'y' else vzR)
    aL = sound_speed(PL, rhoL)
    aR = sound_speed(PR, rhoR)
    sL = np.minimum(vnL - aL, vnR - aR)
    sR = np.maximum(vnL + aL, vnR + aR)
    sM = (PR - PL + rhoL*vnL*(sL - vnL) - rhoR*vnR*(sR - vnR))/_safe_denom(rhoL*(sL - vnL) - rhoR*(sR - vnR))
    rhoL_star = rhoL*(sL - vnL)/_safe_denom(sL - sM)
    rhoR_star = rhoR*(sR - vnR)/_safe_denom(sR - sM)
    UL_star, pL_star = _star_state(UL, PL, sL, sM, rhoL_star, axis)
    UR_star, pR_star = _star_state(UR, PR, sR, sM, rhoR_star, axis)
    FL = _flux_components(UL, axis)
    FR = _flux_components(UR, axis)
    FL_star = FL + sL[None, ...]*(UL_star - UL)
    FR_star = FR + sR[None, ...]*(UR_star - UR)
    F = np.where((sL >= 0.0)[None, ...], FL,
        np.where((sM >= 0.0)[None, ...], FL_star,
        np.where((sR > 0.0)[None, ...], FR_star, FR)))
    F_hlle, _ = _hlle(UL, UR, axis)
    bad = (rhoL_star <= 0.0) | (rhoR_star <= 0.0) | (pL_star <= 0.0) | (pR_star <= 0.0) | ~np.all(np.isfinite(F), axis=0)
    F = np.where(bad[None, ...], F_hlle, F)
    return F, F[0]


def _transverse_neighbor_max(A, axes):
    out = A.copy()
    for axis in axes:
        lo = [slice(None)] * A.ndim
        hi = [slice(None)] * A.ndim
        lo[axis] = slice(1, None)
        hi[axis] = slice(None, -1)
        out[tuple(lo)] = np.maximum(out[tuple(lo)], A[tuple(hi)])
        out[tuple(hi)] = np.maximum(out[tuple(hi)], A[tuple(lo)])
    return out


def _multidimensional_shock_sensor(UL, UR, axis):
    """Pressure/compression sensor on the transverse H stencil."""
    from .. import settings as S
    rhoL, vxL, vyL, vzL, PL = prim_from_cons(UL)
    rhoR, vxR, vyR, vzR, PR = prim_from_cons(UR)
    vnL = vxL if axis == 'x' else (vyL if axis == 'y' else vzL)
    vnR = vxR if axis == 'x' else (vyR if axis == 'y' else vzR)
    aL = sound_speed(PL, rhoL)
    aR = sound_speed(PR, rhoR)
    pjump = np.abs(PR - PL)/np.maximum(PR + PL, 1.0e-30)
    compression = np.maximum(vnL - vnR, 0.0)/np.maximum(aL + aR, 1.0e-30)
    p0 = np.clip(float(getattr(S, 'SHOCK_PRESSURE_THRESHOLD', 0.2)), 0.0, 1.0)
    p1 = np.clip(float(getattr(S, 'SHOCK_FULL_HLLE_THRESHOLD', 0.6)), p0 + 1.0e-12, 1.0)
    c0 = float(getattr(S, 'SHOCK_COMPRESSION_THRESHOLD', 0.05))
    sensor = np.clip((pjump - p0)/max(p1 - p0, 1.0e-12), 0.0, 1.0)
    sensor *= np.clip(compression/max(c0, 1.0e-12), 0.0, 1.0)
    transverse = tuple(range(sensor.ndim)) if sensor.ndim < 3 else {'x': (1, 2), 'y': (0, 2), 'z': (0, 1)}[axis]
    return _transverse_neighbor_max(sensor, transverse)


def _riemann(UL, UR, axis):
    """Dispatch to selected Riemann solver based on S.RIEMANN."""
    from .. import settings as S
    choice = str(getattr(S, 'RIEMANN', 'hlle')).lower()
    if choice == 'hllc':
        F, _ = _hllc(UL, UR, axis)
        if bool(getattr(S, 'CARBUNCLE_FIX_ON', True)):
            F_hlle, _ = _hlle(UL, UR, axis)
            blend = float(getattr(S, 'CARBUNCLE_HLLE_BLEND', 1.0))*_multidimensional_shock_sensor(UL, UR, axis)
            blend = np.clip(blend, 0.0, 1.0)
            F = (1.0 - blend[None, ...])*F + blend[None, ...]*F_hlle
        return F, F[0]
    if choice == 'hlle': return _hlle(UL, UR, axis)
    raise ValueError("RIEMANN must be 'hlle' or 'hllc'")

def _face_fluxes(U, dt=0.0, return_predictor=False):
    from .. import settings as S
    DIM3D = bool(getattr(S, 'DIM3D', False))
    """
    Compute vector and mass fluxes on all cell faces.
    Returns:
      Fx_vec (Nvar,Nx+1,Ny,Nz), Fy_vec (Nvar,Nx,Ny+1,Nz), Fz_vec (Nvar,Nx,Ny,Nz+1),
      Mx (Nx+1,Ny,Nz), My (Nx,Ny+1,Nz), Mz (Nx,Ny,Nz+1)
    """
    Nx_, Ny_, Nz_ = U.shape[1:4]

    order = str(getattr(S, 'ORDER', 'plm')).lower()
    if order not in {'first', 'plm'}:
        raise ValueError("ORDER must be 'first' or 'plm'")
    if order == 'plm':
        q, slopes = _primitive_slopes(U)
        U_half = _hancock_predictor(U, dt, q, slopes) if dt > 0.0 else U
        if dt > 0.0:
            U_half = apply_bc(U_half)

    # -- X faces --
    UL, UR = _reconstruct_prims_plm(U_half, slopes, 'x') if order == 'plm' else _reconstruct_prims_first(U, 'x')
    if order == 'plm': exchange_right_face_state_x(UR, face_axis=1, tag=31)
    Fx_mid, Mx_mid = _riemann(UL, UR, 'x')                # (5,Nx-1,Ny,Nz)
    synchronize_left_owned_flux_x(Fx_mid, face_axis=1, left_index=0, right_index=-1, tag=32)
    Fx_vec = np.zeros((U.shape[0], Nx_+1, Ny_, Nz_), dtype=U.dtype)
    Fx_vec[:, 1:Nx_, :, :] = Fx_mid
    # boundaries: use physical flux F(U) (zero-gradient), then override if wind face
    Fx_vec[:, 0, :, :]  = _flux_components(U[:, 0, :, :], 'x')
    Fx_vec[:, -1, :, :] = _flux_components(U[:, -1, :, :], 'x')
    Mx = Fx_vec[0]

    # -- Y faces --
    UL, UR = _reconstruct_prims_plm(U_half, slopes, 'y') if order == 'plm' else _reconstruct_prims_first(U, 'y')
    Fy_mid, My_mid = _riemann(UL, UR, 'y')                # (5,Nx,Ny-1,Nz)
    Fy_vec = np.zeros((U.shape[0], Nx_, Ny_+1, Nz_), dtype=U.dtype)
    Fy_vec[:, :, 1:Ny_, :] = Fy_mid
    Fy_vec[:, :, 0, :]  = _flux_components(U[:, :, 0, :], 'y')
    Fy_vec[:, :, -1, :] = _flux_components(U[:, :, -1, :], 'y')
    My = Fy_vec[0]

    # -- Z faces --
    Fz_vec = np.zeros((U.shape[0], Nx_, Ny_, Nz_+1), dtype=U.dtype)
    if DIM3D:
        UL, UR = _reconstruct_prims_plm(U_half, slopes, 'z') if order == 'plm' else _reconstruct_prims_first(U, 'z')
        Fz_mid, Mz_mid = _riemann(UL, UR, 'z')            # (5,Nx,Ny,Nz-1)
        Fz_vec[:, :, :, 1:Nz_] = Fz_mid
        Fz_vec[:, :, :, 0]  = _flux_components(U[:, :, :, 0], 'z')
        Fz_vec[:, :, :, -1] = _flux_components(U[:, :, :, -1], 'z')
    Mz = Fz_vec[0]

    face, U_in = _boundary_inflow_state()
    if U_in is not None:
        if face == 'x-' and RANK == 0:
            UL = U_in[:, None, None]
            UR = U[:, 0, :, :]
            F, M = _riemann(UL, UR, 'x')
            Fx_vec[:, 0, :, :] = F
            Mx[0, :, :] = M
        elif face == 'x+' and RANK == SIZE - 1:
            UL = U[:, -1, :, :]
            UR = U_in[:, None, None]
            F, M = _riemann(UL, UR, 'x')
            Fx_vec[:, -1, :, :] = F
            Mx[-1, :, :] = M
        elif face == 'y-':
            UL = U_in[:, None, None]
            UR = U[:, :, 0, :]
            F, M = _riemann(UL, UR, 'y')
            Fy_vec[:, :, 0, :] = F
            My[:, 0, :] = M
        elif face == 'y+':
            UL = U[:, :, -1, :]
            UR = U_in[:, None, None]
            F, M = _riemann(UL, UR, 'y')
            Fy_vec[:, :, -1, :] = F
            My[:, -1, :] = M
        elif face == 'z-' and DIM3D:
            UL = U_in[:, None, None]
            UR = U[:, :, :, 0]
            F, M = _riemann(UL, UR, 'z')
            Fz_vec[:, :, :, 0] = F
            Mz[:, :, 0] = M
        elif face == 'z+' and DIM3D:
            UL = U[:, :, :, -1]
            UR = U_in[:, None, None]
            F, M = _riemann(UL, UR, 'z')
            Fz_vec[:, :, :, -1] = F
            Mz[:, :, -1] = M

    result = (Fx_vec, Fy_vec, Fz_vec, Mx, My, Mz)
    return result + ((U_half if order == 'plm' else U),) if return_predictor else result


# --------------------------- UPDATE / BOUNDARIES ----------------------------
def apply_bc(U):
    from .. import settings as S
    DIM3D = bool(getattr(S, 'DIM3D', False))
    # Use MPI-aware copy BCs in X, plain copy in Y/Z
    U = apply_bc_mpi(U, physical_left=(RANK==0), physical_right=(RANK==SIZE-1))
    exchange_ghosts_x(U)
    face, U_in = _boundary_inflow_state()
    if U_in is not None:
        if face == 'x-' and RANK == 0:
            U[:, 0, :, :] = U_in[:, None, None]
        elif face == 'x+' and RANK == SIZE - 1:
            U[:, -1, :, :] = U_in[:, None, None]
        elif face == 'y-':
            U[:, :, 0, :] = U_in[:, None, None]
        elif face == 'y+':
            U[:, :, -1, :] = U_in[:, None, None]
        elif face == 'z-' and DIM3D:
            U[:, :, :, 0] = U_in[:, None, None]
        elif face == 'z+' and DIM3D:
            U[:, :, :, -1] = U_in[:, None, None]
    return U


def apply_tracer_bc(D):
    from .. import settings as S
    apply_bc_mpi(D[None, ...], physical_left=(RANK==0), physical_right=(RANK==SIZE-1))
    exchange_ghosts_x_scalar(D)
    face, _ = _boundary_inflow_state()
    if face == 'x-' and RANK == 0: D[0, :, :] = 0.0
    elif face == 'x+' and RANK == SIZE - 1: D[-1, :, :] = 0.0
    elif face == 'y-': D[:, 0, :] = 0.0
    elif face == 'y+': D[:, -1, :] = 0.0
    elif face == 'z-' and bool(getattr(S, 'DIM3D', False)): D[:, :, 0] = 0.0
    elif face == 'z+' and bool(getattr(S, 'DIM3D', False)): D[:, :, -1] = 0.0
    return D


def cfl_dt(U):
    from .. import settings as S
    dx = getattr(S, 'dx', None)
    dy = getattr(S, 'dy', dx)
    dz = getattr(S, 'dz', dx)
    CFL = getattr(S, 'CFL', 0.4)
    DIM3D = bool(getattr(S, 'DIM3D', False))
    from .. import settings as S
    """Global CFL time step using max(|v|+a) in each direction."""
    Ui = U[:, 1:-1, :, :] if U.ndim == 4 and U.shape[1] > 2 else U
    rho, vx, vy, vz, P = prim_from_cons(Ui)
    a = sound_speed(P, rho)
    vmax_x = float(np.max(np.abs(vx) + a))
    vmax_y = float(np.max(np.abs(vy) + a))
    vmax_z = float(np.max(np.abs(vz) + a)) if S.DIM3D else 0.0
    dt_x = dx / vmax_x if vmax_x > 0.0 else 1e99
    dt_y = dy / vmax_y if vmax_y > 0.0 else 1e99
    dt_z = dz / vmax_z if S.DIM3D and vmax_z > 0.0 else 1e99
    dt_local = CFL * min(dt_x, dt_y, dt_z)
    try:
        from mpi4py import MPI
        return COMM.allreduce(dt_local, op=MPI.MIN)
    except Exception:
        return COMM.allreduce(dt_local)


def step_euler(U, dt, return_fluxes=False):
    from .. import settings as S
    dx = getattr(S, 'dx', None)
    dy = getattr(S, 'dy', dx)
    dz = getattr(S, 'dz', dx)
    DIM3D = bool(getattr(S, 'DIM3D', False))
    """
    Finite-volume update; PLM uses MUSCL-Hancock and first order uses Godunov-Euler.
    """
    # Enforce BCs (so boundary cells are valid for face fluxes)
    U = apply_bc(U)

    # Face fluxes
    Fx_vec, Fy_vec, Fz_vec, Mx, My, Mz, U_source = _face_fluxes(U, dt, return_predictor=True)

    # Divergence update
    U_new = U.copy()
    # x
    U_new[:, :, :, :] -= (dt/dx) * (Fx_vec[:, 1:, :, :] - Fx_vec[:, :-1, :, :])
    # y
    U_new[:, :, :, :] -= (dt/dy) * (Fy_vec[:, :, 1:, :] - Fy_vec[:, :, :-1, :])
    # z
    if DIM3D:
        U_new[:, :, :, :] -= (dt/dz) * (Fz_vec[:, :, :, 1:] - Fz_vec[:, :, :, :-1])

    if U_new.shape[0] > 5:
        _, vx, vy, vz, P = prim_from_cons(U_source)
        divv = np.gradient(vx, dx, axis=0) + np.gradient(vy, dy, axis=1)
        if DIM3D: divv += np.gradient(vz, dz, axis=2)
        U_new[5] -= dt*P*divv

    # Dual-energy fallback for under-resolved thermal energy.
    U_new = eos.reconcile_dual_energy(U_new)

    # Apply BCs after update
    U_new = apply_bc_mpi(U_new, physical_left=(RANK==0), physical_right=(RANK==SIZE-1))

    # Reset scalar-call counter (used by advect_scalar_consistent)
    global _WIND_SCALAR_CALL_COUNT

    _WIND_SCALAR_CALL_COUNT = 0

    if return_fluxes:
        return U_new, (Mx, My, Mz)
    return U_new


# Expose mass-face fluxes to main fallback if needed
def _face_mass_fluxes(U, dt=0.0):
    """Return only the mass fluxes (ρv·n) at faces."""
    _, _, _, Mx, My, Mz = _face_fluxes(U, dt)
    return Mx, My, Mz


# --------------------------- SCALAR ADVECTION -------------------------------

def advect_scalar_consistent(rho_old, rho_new, D, dt, Fx, Fy, Fz):
    from .. import settings as S
    dx = getattr(S, 'dx', None)
    dy = getattr(S, 'dy', dx)
    dz = getattr(S, 'dz', dx)
    DIM3D = bool(getattr(S, 'DIM3D', False))
    Q = rho_old * D
    Nx_, Ny_, Nz_ = Q.shape
    def faces(Q, rho, M, axis):
        import numpy as np
        if axis=='x':
            sQ = np.zeros_like(Q); sr = np.zeros_like(rho)
            sQ[1:-1,:,:] = _mc(Q[1:-1,:,:]-Q[0:-2,:,:], Q[2:,:,:]-Q[1:-1,:,:])
            sr[1:-1,:,:] = _mc(rho[1:-1,:,:]-rho[0:-2,:,:], rho[2:,:,:]-rho[1:-1,:,:])
            QL = Q[0:-1,:,:] + 0.5*sQ[0:-1,:,:]; QR = Q[1:,:,:] - 0.5*sQ[1:,:,:]
            rL = rho[0:-1,:,:] + 0.5*sr[0:-1,:,:]; rR = rho[1:,:,:] - 0.5*sr[1:,:,:]
            up = M[1:Nx_,:,:] >= 0.0
            DL = _buf_like('DL', QL); DL.fill(0); np.divide(QL, rL, out=DL, where=(rL>0.0)); DR = _buf_like('DR', QR); DR.fill(0); np.divide(QR, rR, out=DR, where=(rR>0.0))
            exchange_right_face_state_x(DR, face_axis=0, tag=41)
            Df = np.where(up, DL, DR)
            F = np.zeros((Nx_+1, Ny_, Nz_), dtype=Q.dtype)
            F[1:Nx_,:,:] = M[1:Nx_,:,:]*Df; F[0,:,:]=M[0,:,:]*D[0,:,:]; F[Nx_,:,:]=M[Nx_,:,:]*D[-1,:,:]
            synchronize_left_owned_flux_x(F, face_axis=0, left_index=1, right_index=-2, tag=42)
            return F
        if axis=='y':
            sQ = np.zeros_like(Q); sr = np.zeros_like(rho)
            sQ[:,1:-1,:] = _mc(Q[:,1:-1,:]-Q[:,0:-2,:], Q[:,2:,:]-Q[:,1:-1,:])
            sr[:,1:-1,:] = _mc(rho[:,1:-1,:]-rho[:,0:-2,:], rho[:,2:,:]-rho[:,1:-1,:])
            QL = Q[:,0:-1,:] + 0.5*sQ[:,0:-1,:]; QR = Q[:,1:,:] - 0.5*sQ[:,1:,:]
            rL = rho[:,0:-1,:] + 0.5*sr[:,0:-1,:]; rR = rho[:,1:,:] - 0.5*sr[:,1:,:]
            up = M[:,1:Ny_,:] >= 0.0
            DL = _buf_like('DL', QL); DL.fill(0); np.divide(QL, rL, out=DL, where=(rL>0.0)); DR = _buf_like('DR', QR); DR.fill(0); np.divide(QR, rR, out=DR, where=(rR>0.0))
            Df = np.where(up, DL, DR)
            F = np.zeros((Nx_, Ny_+1, Nz_), dtype=Q.dtype)
            F[:,1:Ny_,:] = M[:,1:Ny_,:]*Df; F[:,0,:]=M[:,0,:]*D[:,0,:]; F[:,Ny_,:]=M[:,Ny_,:]*D[:,-1,:]
            return F
        sQ = np.zeros_like(Q); sr = np.zeros_like(rho)
        sQ[:,:,1:-1] = _mc(Q[:,:,1:-1]-Q[:,:,0:-2], Q[:,:,2:]-Q[:,:,1:-1])
        sr[:,:,1:-1] = _mc(rho[:,:,1:-1]-rho[:,:,0:-2], rho[:,:,2:]-rho[:,:,1:-1])
        QL = Q[:,:,0:-1] + 0.5*sQ[:,:,0:-1]; QR = Q[:,:,1:] - 0.5*sQ[:,:,1:]
        rL = rho[:,:,0:-1] + 0.5*sr[:,:,0:-1]; rR = rho[:,:,1:] - 0.5*sr[:,:,1:]
        up = M[:,:,1:Nz_] >= 0.0
        DL = _buf_like('DL', QL); DL.fill(0); np.divide(QL, rL, out=DL, where=(rL>0.0)); DR = _buf_like('DR', QR); DR.fill(0); np.divide(QR, rR, out=DR, where=(rR>0.0))
        Df = np.where(up, DL, DR)
        F = np.zeros((Nx_, Ny_, Nz_+1), dtype=Q.dtype)
        F[:,:,1:Nz_] = M[:,:,1:Nz_]*Df; F[:,:,0]=M[:,:,0]*D[:,:,0]; F[:,:,Nz_]=M[:,:,Nz_]*D[:,:,-1]
        return F

    FQx = faces(Q, rho_old, Fx, 'x')
    FQy = faces(Q, rho_old, Fy, 'y')
    FQz = faces(Q, rho_old, Fz, 'z') if DIM3D else 0.0

    if DIM3D:
        Q = Q - (dt/dx)*(FQx[1:, :, :] - FQx[:-1, :, :]) - (dt/dy)*(FQy[:, 1:, :] - FQy[:, :-1, :]) - (dt/dz)*(FQz[:, :, 1:] - FQz[:, :, :-1])
    else:
        Q = Q - (dt/dx)*(FQx[1:, :, :] - FQx[:-1, :, :]) - (dt/dy)*(FQy[:, 1:, :] - FQy[:, :-1, :])

    D_new = np.divide(Q, rho_new, out=np.zeros_like(Q), where=(rho_new>0.0))
    return np.clip(D_new, 0.0, 1.0)


def advect_scalar_consistent_multi(rho_old, rho_new, D_bins, dt, Fx, Fy, Fz):
    """
    Vectorized consistent advection for NBINS dust-to-gas ratio scalars.
    D_bins shape: (Nx, Ny, Nz, NBINS) with NBINS as last axis.
    """
    from .. import settings as S
    dx = float(getattr(S, 'dx', None))
    dy = float(getattr(S, 'dy', dx))
    dz = float(getattr(S, 'dz', dx))
    DIM3D = bool(getattr(S, 'DIM3D', False))

    Q = rho_old[..., None] * D_bins  # dust mass per bin
    Nx_, Ny_, Nz_, NB = Q.shape

    def faces_multi(Q, rho, M, axis):
        import numpy as _np
        if axis == 'x':
            sQ = _np.zeros_like(Q); sr = _np.zeros_like(rho)
            sQ[1:-1, ...] = _mc(Q[1:-1, ...] - Q[0:-2, ...], Q[2:, ...] - Q[1:-1, ...])
            sr[1:-1, ...] = _mc(rho[1:-1, ...] - rho[0:-2, ...], rho[2:, ...] - rho[1:-1, ...])
            QL = Q[0:-1, ...] + 0.5 * sQ[0:-1, ...]
            QR = Q[1:,   ...] - 0.5 * sQ[1:,   ...]
            rL = rho[0:-1, ...] + 0.5 * sr[0:-1, ...]
            rR = rho[1:,   ...] - 0.5 * sr[1:,   ...]
            up = (M[1:Nx_, :, :] >= 0.0)[..., None]
            DL = _buf_like('DLm', QL); DL.fill(0.0); _np.divide(QL, rL[..., None], out=DL, where=(rL > 0.0)[..., None])
            DR = _buf_like('DRm', QR); DR.fill(0.0); _np.divide(QR, rR[..., None], out=DR, where=(rR > 0.0)[..., None])
            exchange_right_face_state_x(DR, face_axis=0, tag=43)
            Df = _np.where(up, DL, DR)
            F = _np.zeros((Nx_+1, Ny_, Nz_, NB), dtype=Q.dtype)
            F[1:Nx_, ...] = M[1:Nx_, :, :][..., None] * Df
            F[0,     ...] = M[0,     :, :][..., None] * D_bins[0,     ...]
            F[Nx_,   ...] = M[Nx_,   :, :][..., None] * D_bins[-1,    ...]
            synchronize_left_owned_flux_x(F, face_axis=0, left_index=1, right_index=-2, tag=44)
            return F
        if axis == 'y':
            sQ = _np.zeros_like(Q); sr = _np.zeros_like(rho)
            sQ[:,1:-1, ...] = _mc(Q[:,1:-1, ...] - Q[:,0:-2, ...], Q[:,2:, ...] - Q[:,1:-1, ...])
            sr[:,1:-1, ...] = _mc(rho[:,1:-1, ...] - rho[:,0:-2, ...], rho[:,2:, ...] - rho[:,1:-1, ...])
            QL = Q[:,0:-1, ...] + 0.5 * sQ[:,0:-1, ...]
            QR = Q[:,1:,   ...] - 0.5 * sQ[:,1:,   ...]
            rL = rho[:,0:-1, ...] + 0.5 * sr[:,0:-1, ...]
            rR = rho[:,1:,   ...] - 0.5 * sr[:,1:,   ...]
            up = (M[:,1:Ny_, : ] >= 0.0)[..., None]
            DL = _buf_like('DLy', QL); DL.fill(0.0); _np.divide(QL, rL[..., None], out=DL, where=(rL > 0.0)[..., None])
            DR = _buf_like('DRy', QR); DR.fill(0.0); _np.divide(QR, rR[..., None], out=DR, where=(rR > 0.0)[..., None])
            Df = _np.where(up, DL, DR)
            F = _np.zeros((Nx_, Ny_+1, Nz_, NB), dtype=Q.dtype)
            F[:,1:Ny_, ...] = M[:,1:Ny_, :][..., None] * Df
            F[:,0,     ...] = M[:,0,     :][..., None] * D_bins[:,0,     ...]
            F[:,Ny_,   ...] = M[:,Ny_,   :][..., None] * D_bins[:,-1,    ...]
            return F
        # axis == 'z'
        sQ = _np.zeros_like(Q); sr = _np.zeros_like(rho)
        sQ[:,:,1:-1, ...] = _mc(Q[:,:,1:-1, ...] - Q[:,:,0:-2, ...], Q[:,:,2:, ...] - Q[:,:,1:-1, ...])
        sr[:,:,1:-1, ...] = _mc(rho[:,:,1:-1, ...] - rho[:,:,0:-2, ...], rho[:,:,2:, ...] - rho[:,:,1:-1, ...])
        QL = Q[:,:,0:-1, ...] + 0.5 * sQ[:,:,0:-1, ...]
        QR = Q[:,:,1:,   ...] - 0.5 * sQ[:,:,1:,   ...]
        rL = rho[:,:,0:-1, ...] + 0.5 * sr[:,:,0:-1, ...]
        rR = rho[:,:,1:,   ...] - 0.5 * sr[:,:,1:,   ...]
        up = (M[:,:,1:Nz_] >= 0.0)[..., None]
        DL = _buf_like('DLz', QL); DL.fill(0.0); _np.divide(QL, rL[..., None], out=DL, where=(rL > 0.0)[..., None])
        DR = _buf_like('DRz', QR); DR.fill(0.0); _np.divide(QR, rR[..., None], out=DR, where=(rR > 0.0)[..., None])
        Df = _np.where(up, DL, DR)
        F = _np.zeros((Nx_, Ny_, Nz_+1, NB), dtype=Q.dtype)
        F[:,:,1:Nz_, ...] = M[:,:,1:Nz_][..., None] * Df
        F[:,:,0,     ...] = M[:,:,0    ][..., None] * D_bins[:,:,0,     ...]
        F[:,:,Nz_,   ...] = M[:,:,Nz_  ][..., None] * D_bins[:,:,-1,     ...]
        return F

    FQx = faces_multi(Q, rho_old, Fx, 'x')
    FQy = faces_multi(Q, rho_old, Fy, 'y')
    FQz = faces_multi(Q, rho_old, Fz, 'z') if DIM3D else 0.0

    if DIM3D:
        Q = Q - (dt/dx) * (FQx[1:, ...] - FQx[0:-1, ...]) \
              - (dt/dy) * (FQy[:,1:, ...] - FQy[:,0:-1, ...]) \
              - (dt/dz) * (FQz[:,:,1:, ...] - FQz[:,:,0:-1, ...])
    else:
        Q = Q - (dt/dx) * (FQx[1:, ...] - FQx[0:-1, ...]) \
              - (dt/dy) * (FQy[:,1:, ...] - FQy[:,0:-1, ...])

    D_new = np.zeros_like(D_bins)
    np.divide(Q, rho_new[..., None], out=D_new, where=(rho_new > 0.0)[..., None])
    return np.clip(D_new, 0.0, None)
