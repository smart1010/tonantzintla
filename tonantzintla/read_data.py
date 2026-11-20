# -*- coding: utf-8 -*-
"""
read_data.py: Parse tonantzintla.par (ConfigParser/INI) into a Config dataclass
and emit a self-contained `settings.py` that the rest of the code imports.

Public API used by main.py:
  - cfg = load_config(par_path)
  - source = render_settings_module(cfg)
  - write_settings_module(cfg, path_to_settings_py)

Includes a minimal wind-tunnel section:
  [wind]
  WIND_ON = True/False
  WIND_FACE = 'x-' | 'x+' | 'y-' | 'y+' | 'z-' | 'z+'
  WIND_VX_KMS, WIND_VY_KMS, WIND_VZ_KMS
  WIND_T, WIND_N_CM3
  WIND_DUST_TO_GAS
and writes corresponding constants to settings.py.
"""

import ast
from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Tuple, Optional

def _populate_grid_from_lref_cp(cp):
    """
    Ensure Nx, Ny, Nz exist in [grid] if lref is provided.
    N = 2**(lref+2). Keys are case-sensitive (optionxform=str).
    """
    sec = "grid"
    if not hasattr(cp, "has_section"):
        return
    if not cp.has_section(sec):
        return
    items = dict(cp.items(sec))
    lkey = None
    for k in list(items.keys()):
        if k.lower() == "lref":
            lkey = k
            break
    if lkey is None:
        return
    try:
        lref_val = int(float(items[lkey]))
    except Exception:
        lref_val = int(items[lkey])
    N = 2 ** (lref_val + 2)
    for axis in ("Nx","Ny","Nz"):
        if not cp.has_option(sec, axis):
            cp.set(sec, axis, str(N))



# ------------------------ dataclasses ------------------------

@dataclass
class Run:
    DIM3D: bool
    INTEGRATED_MAPS: bool
    COOLING_ON: bool
    DUST_ON: bool
    TRACERS_ON: bool
    TURB_ON: bool
    TURB_SEED: int
    VIEW_PLANE: str
    T_END_YR: float
    CFL: float
    SNAP_EVERY: int
    TSERIES_EVERY: int
    # optional / defaults
    RESTART: bool = False
    RESTART_SNAP: int = 0
    SNAP_NEXT_OVERRIDE: int = -1
    CHK_EVERY_STEPS: int = 0
    CHK_EVERY_SECONDS: float = 0.0


@dataclass
class Grid:
    Nx: int
    Ny: int
    Nz: int
    XMIN_PC: float
    XMAX_PC: float
    YMIN_PC: float
    YMAX_PC: float
    ZMIN_PC: float
    ZMAX_PC: float


@dataclass
class Ambient:
    RHO_AMB: float
    T_AMB: float
    VX_AMB_KMS: float
    VY_AMB_KMS: float
    VZ_AMB_KMS: float


@dataclass
class Cloud:
    CENTER_PC: Tuple[float, float, float]
    RADIUS_PC: float
    RHO: float
    T: float
    VX_KMS: float
    VY_KMS: float
    VZ_KMS: float


@dataclass
class Turbulence:
    SIGMA_T_AMB_KMS: float
    SIGMA_T_C1_KMS: float
    SIGMA_T_C2_KMS: float


@dataclass
class Cooling:
    COOLING_ON: bool
    DUST_COOL_SKIP_FRAC: float = 1e-3
    DUST_MIN_RATIO: float = 1e-8
    USE_COOLZ: bool = False
    Z_OVER_ZSUN: float = 1.0
    SCHURE_ION_DEG_INDEX: int = 0


@dataclass
class Dust:
    # initial mass fractions for ambient/clouds (dimensionless)
    D_INIT_AMBIENT: float
    D_INIT_CLOUD1: float
    D_INIT_CLOUD2: float
    # gas-dust cooling switch
    DUST_GAS_COOLING_ON: bool
    # binning controls
    DUST_BINS: str                 # 'MRN' | 'lognormal'
    DUST_NBINS: int
    DUST_A_MIN_UM: float
    DUST_A_MAX_UM: float
    LOGNORM_A0_UM: float
    LOGNORM_SIGMA: float
    MRN_Q: float


@dataclass
class Solver:
    ORDER: Optional[str] = None
    RIEMANN: Optional[str] = None
    CARBUNCLE_FIX_ON: bool = True
    SHOCK_PRESSURE_THRESHOLD: float = 0.2
    SHOCK_FULL_HLLE_THRESHOLD: float = 0.6
    SHOCK_COMPRESSION_THRESHOLD: float = 0.05
    CARBUNCLE_HLLE_BLEND: float = 1.0
    HANCOCK_FLATTEN_FRACTION: float = 0.1


@dataclass
class Parallel:
    NPROC_X: Optional[int] = None
    NPROC_Y: Optional[int] = None
    NPROC_Z: Optional[int] = None


@dataclass
class Plot:
    VIEW_PLANE: str
    OUTDIR: str
    BASEN: str
    SNAP_PLOTS_ON: bool = True
    DIAG_PLOTS_ON: bool = True
    LOG_FILE: Optional[str] = None


@dataclass
class Interfaces:
    RAMP_MODE: str = "auto"
    RAMP_CELLS: int = 2
    RAMP_MIN_CELLS: int = 2
    RAMP_MAX_CELLS: int = 16
    RAMP_MAX_DEX_PER_CELL: float = 1.0


@dataclass
class Files:
    LAMBDA_TABLE_FILE: str


# -------- wind inflow controls --------
@dataclass
class Wind:
    ON: bool
    FACE: str               # 'x-','x+','y-','y+','z-','z+'
    VX_KMS: float
    VY_KMS: float
    VZ_KMS: float
    T: float                # K
    N_CM3: float            # total number density [cm^-3]
    DUST_TO_GAS: float      # injected dust-to-gas mass ratio (distributed by bins.mass_fracs)


@dataclass
class Frame:
    ON: bool
    AXIS: str
    V_KMS: float


@dataclass
class Config:
    run: Run
    grid: Grid
    ambient: Ambient
    cloud1: Cloud
    cloud2: Cloud
    turbulence: Turbulence
    cooling: Cooling
    dust: Dust
    solver: Solver
    parallel: Parallel
    plot: Plot
    interfaces: Interfaces
    files: Files
    wind: Wind
    frame: Frame


# ------------------------ parsing helpers ------------------------

def _lit(raw: str):
    """Literal-evaluate a string when possible, fallback to stripped string."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip()


def _get_str_or_lit(cp: ConfigParser, sec: str, key: str, default=None):
    if not cp.has_section(sec) or not cp.has_option(sec, key):
        return default
    raw = cp.get(sec, key)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip()


# ------------------------ public API: read & assemble ------------------------

def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"parameters file not found: {p}")

    cp = ConfigParser(inline_comment_prefixes=('#',';'), interpolation=None, strict=False)
    cp.optionxform = str  # preserve case of keys
    cp.read(p)

    _populate_grid_from_lref_cp(cp)
    def g(sect, key, default=None):
        # Tolerant getter with lref support and optional default.
        if not cp.has_section(sect):
            if default is not None:
                return default
            raise KeyError(f"Missing required section [{sect}]")

        if not cp.has_option(sect, key):
            if sect == "grid" and key in ("Nx","Ny","Nz"):
                _populate_grid_from_lref_cp(cp)
            if not cp.has_option(sect, key):
                if default is not None:
                    return default
                if sect == "run" and key == "DIM3D":
                    return True
                raise KeyError(f"Missing required key [{sect}] {key}")

        raw = cp.get(sect, key).strip()

        # Helper casters
        def _to_bool(x: str):
            xl = x.lower()
            if xl in ("1","true","t","yes","y","on"): return True
            if xl in ("0","false","f","no","n","off"): return False
            raise ValueError

        # If default provided, cast to that type
        if default is not None:
            t = type(default)
            try:
                if t is bool:  return _to_bool(raw)
                if t is int:   return int(raw)
                if t is float: return float(raw)
                if t is str:
                    try:
                        import ast as _ast
                        return str(_ast.literal_eval(raw))
                    except Exception:
                        return raw
                return t(raw)
            except Exception:
                try:
                    import ast as _ast
                    return _ast.literal_eval(raw)
                except Exception:
                    return default

        # Parsing precedence: literal, bool/int/float, string.
        try:
            import ast as _ast
            return _ast.literal_eval(raw)
        except Exception:
            pass
        for caster in (_to_bool, int, float):
            try:
                return caster(raw)
            except Exception:
                pass
        return raw

    def g_basen(default="run"):
        """Try a few common key names/sections to determine BASEN."""
        cand_keys = ("BASEN", "basen", "basenm", "base", "basename")
        cand_secs = ("plot", "run")
        for sec in cand_secs:
            if cp.has_section(sec):
                for k in cand_keys:
                    val = _get_str_or_lit(cp, sec, k, default=None)
                    if val is not None:
                        return str(val)
        for sec in cp.sections():
            for k in cand_keys:
                val = _get_str_or_lit(cp, sec, k, default=None)
                if val is not None:
                    return str(val)
        return default

    def g_logfile(default=None):
        """Discover an explicit log file name if provided; returns None if not set."""
        cand_keys = ("LOG_FILE", "LOGFILE", "log_file")
        cand_secs = ("plot", "run")
        for sec in cand_secs:
            if cp.has_section(sec):
                for k in cand_keys:
                    val = _get_str_or_lit(cp, sec, k, default=None)
                    if val:
                        return str(val)
        return default

    # ---- Sections ----

    run = Run(
        DIM3D=g("run", "DIM3D"),
        INTEGRATED_MAPS=g("run", "INTEGRATED_MAPS"),
        COOLING_ON=cp.getboolean('run','COOLING_ON', fallback=None),
        DUST_ON=g("run", "DUST_ON"),
        TRACERS_ON=g("run", "TRACERS_ON"),
        TURB_ON=g("run", "TURB_ON"),
        TURB_SEED=g("run", "TURB_SEED"),
        VIEW_PLANE=g("run", "VIEW_PLANE"),
        T_END_YR=g("run", "T_END_YR"),
        CFL=g("run", "CFL"),
        SNAP_EVERY=g("run", "SNAP_EVERY"),
        TSERIES_EVERY=g("run", "TSERIES_EVERY"),
        RESTART=g("run", "RESTART", False),
        RESTART_SNAP=g("run", "RESTART_SNAP", 0),
        SNAP_NEXT_OVERRIDE = g("run", "SNAP_NEXT_OVERRIDE", -1),
        CHK_EVERY_STEPS=g("run", "CHK_EVERY_STEPS", 0),
        CHK_EVERY_SECONDS=g("run", "CHK_EVERY_SECONDS", 0.0),
    )

    grid = Grid(
        Nx=g("grid", "Nx"), Ny=g("grid", "Ny"), Nz=g("grid", "Nz"),
        XMIN_PC=g("grid", "XMIN_PC"), XMAX_PC=g("grid", "XMAX_PC"),
        YMIN_PC=g("grid", "YMIN_PC"), YMAX_PC=g("grid", "YMAX_PC"),
        ZMIN_PC=g("grid", "ZMIN_PC"), ZMAX_PC=g("grid", "ZMAX_PC"),
    )

    ambient = Ambient(
        RHO_AMB=g("ambient", "RHO_AMB"),
        T_AMB=g("ambient", "T_AMB"),
        VX_AMB_KMS=g("ambient", "VX_AMB_KMS"),
        VY_AMB_KMS=g("ambient", "VY_AMB_KMS"),
        VZ_AMB_KMS=g("ambient", "VZ_AMB_KMS"),
    )

    cloud1 = Cloud(
        CENTER_PC=tuple(g("cloud1", "C1_CENTER_PC")),
        RADIUS_PC=g("cloud1", "C1_RADIUS_PC"),
        RHO=g("cloud1", "C1_RHO"),
        T=g("cloud1", "C1_T"),
        VX_KMS=g("cloud1", "C1_VX_KMS"),
        VY_KMS=g("cloud1", "C1_VY_KMS"),
        VZ_KMS=g("cloud1", "C1_VZ_KMS"),
    )

    cloud2 = Cloud(
        CENTER_PC=tuple(g("cloud2", "C2_CENTER_PC")),
        RADIUS_PC=g("cloud2", "C2_RADIUS_PC"),
        RHO=g("cloud2", "C2_RHO"),
        T=g("cloud2", "C2_T"),
        VX_KMS=g("cloud2", "C2_VX_KMS"),
        VY_KMS=g("cloud2", "C2_VY_KMS"),
        VZ_KMS=g("cloud2", "C2_VZ_KMS"),
    )

    turbulence = Turbulence(
        SIGMA_T_AMB_KMS=g("turbulence", "SIGMA_T_AMB_KMS"),
        SIGMA_T_C1_KMS=g("turbulence", "SIGMA_T_C1_KMS"),
        SIGMA_T_C2_KMS=g("turbulence", "SIGMA_T_C2_KMS"),
    )

    



    # Resolve COOLING_ON primarily from [cooling]; fallback to [run]
    try:
        COOLING_ON_c = cp.getboolean('cooling','COOLING_ON')
    except Exception:
        COOLING_ON_c = None
    try:
        COOLING_ON_r = cp.getboolean('run','COOLING_ON')
    except Exception:
        COOLING_ON_r = None
    COOLING_ON = COOLING_ON_c if (COOLING_ON_c is not None) else (COOLING_ON_r if (COOLING_ON_r is not None) else False)

    cooling = Cooling(
        COOLING_ON=COOLING_ON,
        USE_COOLZ=g('cooling','USE_COOLZ', default=False),
        Z_OVER_ZSUN=cp.getfloat('cooling','Z_OVER_ZSUN', fallback=1.0),
        SCHURE_ION_DEG_INDEX=int(cp.get('cooling','SCHURE_ION_DEG_INDEX', fallback='0')),
    )

    dust = Dust(
        D_INIT_AMBIENT=g("dust", "D_INIT_AMBIENT"),
        D_INIT_CLOUD1=g("dust", "D_INIT_CLOUD1"),
        D_INIT_CLOUD2=g("dust", "D_INIT_CLOUD2"),
        DUST_GAS_COOLING_ON=g("dust", "DUST_GAS_COOLING_ON"),
        DUST_BINS=g("dust", "DUST_BINS"),
        DUST_NBINS=g("dust", "DUST_NBINS"),
        DUST_A_MIN_UM=g("dust", "DUST_A_MIN_UM"),
        DUST_A_MAX_UM=g("dust", "DUST_A_MAX_UM"),
        LOGNORM_A0_UM=g("dust", "LOGNORM_A0_UM"),
        LOGNORM_SIGMA=g("dust", "LOGNORM_SIGMA"),
        MRN_Q=g("dust", "MRN_Q"),
    )

    solver = Solver(
        ORDER=_get_str_or_lit(cp, "solver", "ORDER", None),
        RIEMANN=_get_str_or_lit(cp, "solver", "RIEMANN", None),
        CARBUNCLE_FIX_ON=g("solver", "CARBUNCLE_FIX_ON", True),
        SHOCK_PRESSURE_THRESHOLD=g("solver", "SHOCK_PRESSURE_THRESHOLD", 0.2),
        SHOCK_FULL_HLLE_THRESHOLD=g("solver", "SHOCK_FULL_HLLE_THRESHOLD", 0.6),
        SHOCK_COMPRESSION_THRESHOLD=g("solver", "SHOCK_COMPRESSION_THRESHOLD", 0.05),
        CARBUNCLE_HLLE_BLEND=g("solver", "CARBUNCLE_HLLE_BLEND", 1.0),
        HANCOCK_FLATTEN_FRACTION=g("solver", "HANCOCK_FLATTEN_FRACTION", 0.1),
    )

    parallel = Parallel(
        NPROC_X=_get_str_or_lit(cp, "parallel", "NPROC_X", None),
        NPROC_Y=_get_str_or_lit(cp, "parallel", "NPROC_Y", None),
        NPROC_Z=_get_str_or_lit(cp, "parallel", "NPROC_Z", None),
    )

    plot = Plot(
        VIEW_PLANE=g("plot", "VIEW_PLANE"),
        OUTDIR=g("plot", "OUTDIR", f"snaps_{g_basen('run')}"),
        BASEN=g_basen("run"),
        SNAP_PLOTS_ON=g("plot", "SNAP_PLOTS_ON", True),
        DIAG_PLOTS_ON=g("plot", "DIAG_PLOTS_ON", True),
        LOG_FILE=g_logfile(None),
    )

    interfaces = Interfaces(
        RAMP_MODE=str(g("interfaces", "INTERFACE_RAMP_MODE", "auto")).lower(),
        RAMP_CELLS=g("interfaces", "INTERFACE_RAMP_CELLS", 2),
        RAMP_MIN_CELLS=g("interfaces", "INTERFACE_RAMP_MIN_CELLS", 2),
        RAMP_MAX_CELLS=g("interfaces", "INTERFACE_RAMP_MAX_CELLS", 16),
        RAMP_MAX_DEX_PER_CELL=g("interfaces", "INTERFACE_RAMP_MAX_DEX_PER_CELL", 1.0),
    )

    files = Files(LAMBDA_TABLE_FILE=g("files", "LAMBDA_TABLE_FILE"))

    wind = Wind(
        ON=g("wind", "WIND_ON", False),
        FACE=g("wind", "WIND_FACE", "x-"),
        VX_KMS=g("wind", "WIND_VX_KMS", 777.0),
        VY_KMS=g("wind", "WIND_VY_KMS", 0.0),
        VZ_KMS=g("wind", "WIND_VZ_KMS", 0.0),
        T=g("wind", "WIND_T", 1.0e7),
        N_CM3=g("wind", "WIND_N_CM3", 1.0e-2),
        DUST_TO_GAS=g("wind", "WIND_DUST_TO_GAS", 1.0e-3),
    )

    frame = Frame(
        ON=g("frame", "MOVING_FRAME_ON", False),
        AXIS=str(g("frame", "MOVING_FRAME_AXIS", "z")).lower(),
        V_KMS=g("frame", "MOVING_FRAME_V_KMS", 0.0),
    )

    return Config(run, grid, ambient, cloud1, cloud2,
                  turbulence, cooling, dust, solver, parallel, plot, interfaces, files, wind, frame)


# ------------------------ writer ------------------------

def render_settings_module(cfg: Config):
    """Return the Python source for tonantzintla/settings.py."""
    lines = []

    # Physical constants (cgs)
    lines += [
        '# -- Performance knobs (optional) --',
        '# DUST_COOL_SKIP_FRAC = 1e-3   # skip dust call if upper-bound(L_dust) < frac * L_gas',
        '# DUST_MIN_RATIO = 1e-8        # skip dust call if total D (sum of bins) < this',

        "PC_CM = 3.085677581491367e18",
        "kB    = 1.3806485e-16",
        "mH    = 1.6726219e-24",
        "",
        f"T_AMB   = {cfg.ambient.T_AMB}",
        f"RHO_AMB = {cfg.ambient.RHO_AMB}",
        "",
        "gamma = 5.0/3.0",
        "mu_mol= 2.33*mH",
        "mu_atm= 1.27*mH",
        "mu_ion= 0.61*mH",
        "def _mu_from_T_scalar(T):",
        "    return mu_mol if T <= 300.0 else (mu_mol + (mu_atm-mu_mol)*(T-300.0)/400.0 if T < 700.0 else (mu_atm if T <= 9300.0 else (mu_atm + (mu_ion-mu_atm)*(T-9300.0)/400.0 if T < 9700.0 else mu_ion)))",
        "MU_AMB = _mu_from_T_scalar(T_AMB)",        
        "MSUN  = 1.98847e33",
        "YR_S  = 365.0*24.0*3600.0",
    ]

    # Run controls
    lines += [
        f"DIM3D = {cfg.run.DIM3D!r}",
        f"INTEGRATED_MAPS = {cfg.run.INTEGRATED_MAPS!r}",
        f"COOLING_ON = {bool(cfg.cooling.COOLING_ON)!r}",
        f"DUST_ON = {cfg.run.DUST_ON!r}",
        f"TRACERS_ON = {cfg.run.TRACERS_ON!r}",
        f"TURB_ON = {cfg.run.TURB_ON!r}",
        f"TURB_SEED = {cfg.run.TURB_SEED!r}",
        f"VIEW_PLANE = {cfg.run.VIEW_PLANE!r}",
        f"T_END_YR = {cfg.run.T_END_YR!r}",
        f"CFL = {cfg.run.CFL!r}",
        f"LOG_TIME_EVERY_STEPS = " + str(getattr(cfg.run, "LOG_TIME_EVERY_STEPS", 1)),
        f"SNAP_EVERY = {cfg.run.SNAP_EVERY!r}",
        f"TSERIES_EVERY = {cfg.run.TSERIES_EVERY!r}",
        f"RESTART = {cfg.run.RESTART!r}",
        f"RESTART_SNAP = {cfg.run.RESTART_SNAP!r}",
        f"SNAP_NEXT_OVERRIDE = {cfg.run.SNAP_NEXT_OVERRIDE!r}",
        f"CHK_EVERY_STEPS = {cfg.run.CHK_EVERY_STEPS!r}",
        f"CHK_EVERY_SECONDS = {cfg.run.CHK_EVERY_SECONDS!r}",
    ]

    
    # Cooling controls
    lines += [
    f"COOLING_ON = {bool(cfg.cooling.COOLING_ON)!r}",
    f"USE_COOLZ = {bool(cfg.cooling.USE_COOLZ)!r}",
    f"Z_OVER_ZSUN = {float(cfg.cooling.Z_OVER_ZSUN)!r}",
    f"SCHURE_ION_DEG_INDEX = {int(cfg.cooling.SCHURE_ION_DEG_INDEX)!r}",
]


    # Grid
    lines += [
        f"Nx = {cfg.grid.Nx!r}",
        f"Ny = {cfg.grid.Ny!r}",
        f"Nz = {cfg.grid.Nz!r}",
        f"XMIN_PC = {cfg.grid.XMIN_PC!r}",
        f"XMAX_PC = {cfg.grid.XMAX_PC!r}",
        f"YMIN_PC = {cfg.grid.YMIN_PC!r}",
        f"YMAX_PC = {cfg.grid.YMAX_PC!r}",
        f"ZMIN_PC = {cfg.grid.ZMIN_PC!r}",
        f"ZMAX_PC = {cfg.grid.ZMAX_PC!r}",
    ]

    # Derived grid geometry (cm); REQUIRED by clouds.coordinates()
    lines += [
        "LX = (XMAX_PC - XMIN_PC) * PC_CM",
        "LY = (YMAX_PC - YMIN_PC) * PC_CM",
        "LZ = (ZMAX_PC - ZMIN_PC) * PC_CM",
        "dx = LX / max(Nx, 1)",
        "dy = LY / max(Ny, 1)",
        "dz = LZ / max(Nz, 1)",
        # Volume of a computational cell (use full 3D volume; clouds.py only reads dz if DIM3D)
        "VOL_CELL = dx * dy * (dz if DIM3D else 1.0)",
    ]

    # Ambient
    lines += [
        f"RHO_AMB = {cfg.ambient.RHO_AMB!r}",
        f"T_AMB = {cfg.ambient.T_AMB!r}",
        f"VX_AMB_KMS = {cfg.ambient.VX_AMB_KMS!r}",
        f"VY_AMB_KMS = {cfg.ambient.VY_AMB_KMS!r}",
        f"VZ_AMB_KMS = {cfg.ambient.VZ_AMB_KMS!r}",
        "P_AMB = RHO_AMB * kB * T_AMB / MU_AMB",
    ]

    # Clouds (km/s -> cm/s copies too)
    lines += [
        f"C1_CENTER_PC = {cfg.cloud1.CENTER_PC!r}",
        f"C1_RADIUS_PC = {cfg.cloud1.RADIUS_PC!r}",
        f"C1_RHO = {cfg.cloud1.RHO!r}",
        f"C1_T = {cfg.cloud1.T!r}",
        f"C1_VX_KMS = {cfg.cloud1.VX_KMS!r}",
        f"C1_VY_KMS = {cfg.cloud1.VY_KMS!r}",
        f"C1_VZ_KMS = {cfg.cloud1.VZ_KMS!r}",
        f"C2_CENTER_PC = {cfg.cloud2.CENTER_PC!r}",
        f"C2_RADIUS_PC = {cfg.cloud2.RADIUS_PC!r}",
        f"C2_RHO = {cfg.cloud2.RHO!r}",
        f"C2_T = {cfg.cloud2.T!r}",
        f"C2_VX_KMS = {cfg.cloud2.VX_KMS!r}",
        f"C2_VY_KMS = {cfg.cloud2.VY_KMS!r}",
        f"C2_VZ_KMS = {cfg.cloud2.VZ_KMS!r}",
        f"C1_VX = {cfg.cloud1.VX_KMS*1.0e5!r}",
        f"C1_VY = {cfg.cloud1.VY_KMS*1.0e5!r}",
        f"C1_VZ = {cfg.cloud1.VZ_KMS*1.0e5!r}",
        f"C2_VX = {cfg.cloud2.VX_KMS*1.0e5!r}",
        f"C2_VY = {cfg.cloud2.VY_KMS*1.0e5!r}",
        f"C2_VZ = {cfg.cloud2.VZ_KMS*1.0e5!r}",
    ]

    ramp_mode = str(cfg.interfaces.RAMP_MODE).lower()
    ramp_cells = int(cfg.interfaces.RAMP_CELLS)
    ramp_min = int(cfg.interfaces.RAMP_MIN_CELLS)
    ramp_max = int(cfg.interfaces.RAMP_MAX_CELLS)
    ramp_dex = float(cfg.interfaces.RAMP_MAX_DEX_PER_CELL)
    if ramp_mode not in {"sharp", "fixed", "auto"}:
        raise ValueError("INTERFACE_RAMP_MODE must be 'sharp', 'fixed', or 'auto'")
    if ramp_cells < 1 or ramp_min < 1 or ramp_max < ramp_min or ramp_dex <= 0.0:
        raise ValueError("Interface ramp parameters must be positive and max cells must be >= min cells")
    lines += [
        f"INTERFACE_RAMP_MODE = {ramp_mode!r}",
        f"INTERFACE_RAMP_CELLS = {ramp_cells!r}",
        f"INTERFACE_RAMP_MIN_CELLS = {ramp_min!r}",
        f"INTERFACE_RAMP_MAX_CELLS = {ramp_max!r}",
        f"INTERFACE_RAMP_MAX_DEX_PER_CELL = {ramp_dex!r}",
    ]

    # Wind-tunnel controls (km/s & cgs velocities exposed)
    lines += [
        f"WIND_ON = {cfg.wind.ON!r}",
        f"WIND_FACE = {cfg.wind.FACE!r}",
        f"WIND_VX_KMS = {cfg.wind.VX_KMS!r}",
        f"WIND_VY_KMS = {cfg.wind.VY_KMS!r}",
        f"WIND_VZ_KMS = {cfg.wind.VZ_KMS!r}",
        f"WIND_VX = {cfg.wind.VX_KMS*1.0e5!r}",
        f"WIND_VY = {cfg.wind.VY_KMS*1.0e5!r}",
        f"WIND_VZ = {cfg.wind.VZ_KMS*1.0e5!r}",
        f"WIND_T = {cfg.wind.T!r}",
        f"WIND_N_CM3 = {cfg.wind.N_CM3!r}",
        f"WIND_DUST_TO_GAS = {cfg.wind.DUST_TO_GAS!r}",
        f"MOVING_FRAME_ON = {cfg.frame.ON!r}",
        f"MOVING_FRAME_AXIS = {cfg.frame.AXIS!r}",
        f"MOVING_FRAME_V_KMS = {cfg.frame.V_KMS!r}",
        f"MOVING_FRAME_V = {cfg.frame.V_KMS*1.0e5!r}",
    ]

    # Turbulence + dust controls
    lines += [
        f"SIGMA_T_AMB_KMS = {cfg.turbulence.SIGMA_T_AMB_KMS!r}",
        f"SIGMA_T_C1_KMS = {cfg.turbulence.SIGMA_T_C1_KMS!r}",
        f"SIGMA_T_C2_KMS = {cfg.turbulence.SIGMA_T_C2_KMS!r}",
        f"D_INIT_AMBIENT = {cfg.dust.D_INIT_AMBIENT!r}",
        f"D_INIT_CLOUD1 = {cfg.dust.D_INIT_CLOUD1!r}",
        f"D_INIT_CLOUD2 = {cfg.dust.D_INIT_CLOUD2!r}",
        f"DUST_GAS_COOLING_ON = {cfg.dust.DUST_GAS_COOLING_ON!r}",
        f"DUST_BINS = {cfg.dust.DUST_BINS!r}",
        f"DUST_NBINS = {cfg.dust.DUST_NBINS!r}",
        f"DUST_A_MIN_UM = {cfg.dust.DUST_A_MIN_UM!r}",
        f"DUST_A_MAX_UM = {cfg.dust.DUST_A_MAX_UM!r}",
        f"LOGNORM_A0_UM = {cfg.dust.LOGNORM_A0_UM!r}",
        f"LOGNORM_SIGMA = {cfg.dust.LOGNORM_SIGMA!r}",
        f"MRN_Q = {cfg.dust.MRN_Q!r}",
    ]

    # Files / plotting
    basen = str(cfg.plot.BASEN)
    outdir = str(cfg.plot.OUTDIR)
    log_file_name = cfg.plot.LOG_FILE if cfg.plot.LOG_FILE else f"{basen}.log"

    lines += [
        f"OUTDIR = {outdir!r}",
        f"BASEN = {basen!r}",
        f"SNAP_PLOTS_ON = {bool(cfg.plot.SNAP_PLOTS_ON)!r}",
        f"DIAG_PLOTS_ON = {bool(cfg.plot.DIAG_PLOTS_ON)!r}",
        "LOGFILE = " + repr(f"{outdir}/{log_file_name}"),
        "MASS_FILE = " + repr(f"{outdir}/mass_history.dat"),
        "GROWZONE_FILE = " + repr(f"{outdir}/growth_zone_fractions.dat"),
        "CB_WIDTH_REL = 0.05",
        "CB_PAD_REL = 0.02",
        "CB_ASPECT = 50",
        "CB_FONT = 10",
        "FIG_DPI = 200",
        f"LAMBDA_TABLE_FILE = {cfg.files.LAMBDA_TABLE_FILE!r}",
    ]

    
    # --- Solver selection ---
    _riem = (cfg.solver.RIEMANN or 'hlle').lower() if hasattr(cfg, 'solver') else 'hlle'
    _orde = (cfg.solver.ORDER or 'plm').lower() if hasattr(cfg, 'solver') else 'plm'
    if _riem not in {'hlle', 'hllc'}:
        raise ValueError("RIEMANN must be 'hlle' or 'hllc'")
    if _orde not in {'first', 'plm'}:
        raise ValueError("ORDER must be 'first' or 'plm'")
    p0 = float(cfg.solver.SHOCK_PRESSURE_THRESHOLD)
    p1 = float(cfg.solver.SHOCK_FULL_HLLE_THRESHOLD)
    c0 = float(cfg.solver.SHOCK_COMPRESSION_THRESHOLD)
    blend = float(cfg.solver.CARBUNCLE_HLLE_BLEND)
    flatten = float(cfg.solver.HANCOCK_FLATTEN_FRACTION)
    if not (0.0 <= p0 < p1 <= 1.0):
        raise ValueError('Shock pressure thresholds must satisfy 0 <= start < full <= 1')
    if c0 <= 0.0 or not (0.0 <= blend <= 1.0):
        raise ValueError('Shock compression threshold must be positive and HLLE blend must lie in [0,1]')
    if not (0.0 < flatten <= 1.0):
        raise ValueError('HANCOCK_FLATTEN_FRACTION must lie in (0,1]')
    lines += [
        f"RIEMANN = {_riem!r}",
        f"ORDER = {_orde!r}",
        f"CARBUNCLE_FIX_ON = {bool(cfg.solver.CARBUNCLE_FIX_ON)!r}",
        f"SHOCK_PRESSURE_THRESHOLD = {p0!r}",
        f"SHOCK_FULL_HLLE_THRESHOLD = {p1!r}",
        f"SHOCK_COMPRESSION_THRESHOLD = {c0!r}",
        f"CARBUNCLE_HLLE_BLEND = {blend!r}",
        f"HANCOCK_FLATTEN_FRACTION = {flatten!r}",
    ]

    return "# Auto-generated from tonantzintla.par. Do not edit by hand.\n" + "\n".join(lines) + "\n"


def write_settings_module(cfg: Config, path: str):
    """Write tonantzintla/settings.py atomically and return its source."""
    p = Path(path)
    text = render_settings_module(cfg)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(text)
    tmp.replace(p)
    return text
