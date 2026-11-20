# tonantzintla/tools/plot_cooling.py
# Plot Λ_dust(T) and Λ_gas(T,Z) with prescribed axes: log10 T in [4, 8] and log10 Λ in [-26, -19].
# Python 3.8 compatibility shim for CINDER imports.

import sys
import re
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

# --- Package import path for script execution ---
_HERE = Path(__file__).resolve()
_outer = _HERE.parent            # .../tonantzintla/tools
_root  = _outer.parent           # .../tonantzintla (outer)
if (_root / "tonantzintla").exists() and str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tonantzintla.read_data import load_config, write_settings_module  # noqa: E402
from tonantzintla.cooling.schure_tables import lambda_hd              # noqa: E402
import tonantzintla.cooling as coolpkg                                # noqa: E402

# ---------- Shim importer for Python 3.8 (CINDER uses PEP 604 unions) ----------
def _shim_import_py38(rel_mod_path: Path, qualname: str):
    """
    Load a package module by rewriting PEP 604 unions (A | B) into typing.Union[A, B]
    and X | None into Optional[X]. Registers in sys.modules under `qualname`.
    """
    src_path = _root / "tonantzintla" / rel_mod_path
    if not src_path.exists():
        for cand in (_outer.parent / rel_mod_path, _outer / rel_mod_path):
            if cand.exists():
                src_path = cand
                break
        else:
            raise FileNotFoundError(f"Cannot locate module source: {rel_mod_path}")

    text = src_path.read_text(encoding="utf-8", errors="ignore")

    # Ensure typing imports
    if "from typing import" not in text:
        text = "from typing import Optional, Union\n" + text
    else:
        if "Optional" not in text:
            text = text.replace("from typing import", "from typing import Optional,")
        if "Union" not in text:
            text = text.replace("from typing import", "from typing import Union,")

    # X | None -> Optional[X] (annotations)
    text = re.sub(r":\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*\|\s*None\b",
                  r": Optional[\1]", text)
    text = re.sub(r"->\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*\|\s*None\b",
                  r"-> Optional[\1]", text)
    # A | B -> Union[A, B] (annotations only)
    def _union_repl(m):
        a = m.group(2).strip()
        b = m.group(3).strip()
        return m.group(1) + f"Union[{a}, {b}]"
    text = re.sub(
        r"(?P<pre>[:]|->)\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*\|\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)",
        _union_repl, text
    )

    mod = type(sys)("shimmed_" + qualname)
    mod.__file__ = str(src_path)
    mod.__package__ = ".".join(qualname.split(".")[:-1])
    ns = mod.__dict__
    ns.setdefault("np", np)  # many CINDER files alias numpy as np
    code = compile(text, str(src_path), "exec")
    exec(code, ns)
    sys.modules[qualname] = mod
    return mod

# Shim-import CINDER modules (keep on-disk files untouched)
cm = _shim_import_py38(Path("cinder/cinder_massfrac.py"), "tonantzintla.cinder.cinder_massfrac")
cc = _shim_import_py38(Path("cinder/cinder_cool.py"),     "tonantzintla.cinder.cinder_cool")


# ------------------------------ GAS TABLE UTIL ---------------------------------
def _cooling_tables_paths() -> Tuple[str, str]:
    """Return absolute paths to Schure+09 (>1e4 K) and Dalgarno–McCray72 (<=1e4 K) tables."""
    base = os.path.join(os.path.dirname(coolpkg.__file__), "tables")
    sch_high = os.path.join(base, "cooltab-schure09")
    sch_low  = os.path.join(base, "cooltab-schure09-t4-dalgarno-mccray72")
    return sch_high, sch_low


# ----------------------------------- MAIN --------------------------------------
def main(par_path: str, D_override: Optional[float], save_path: Optional[str]):
    # Apply dust-bin parameters from the configuration.
    cfg = load_config(par_path)
    settings_path = _root / "tonantzintla" / "settings.py"
    write_settings_module(cfg, settings_path)

    # === DUST: total curve via CINDER bins =====================================
    D_plot = float(D_override if D_override is not None else cfg.dust.D_INIT_CLOUD1)
    edges_um, mass_fracs = cm.make_bins()
    a_cm, inv_m = cm.representative_sizes(edges_um)

    # Temperature grid
    logT = np.linspace(4.0, 8.0, 400)
    T    = np.power(10.0, logT)

    # Per-bin kernel Λ_bin(T) (NBINS, NT); total dust curve at D_plot
    K = cc.per_bin_kernel_graphite(T, a_cm, inv_m, Ye=1.0)
    fracs_bins = D_plot * mass_fracs
    Lambda_dust = np.dot(fracs_bins, K)  # erg cm^3 s^-1

    # Sample “cloud1 cell” at its parfile temperature
    T_cell = float(cfg.cloud1.T)
    K_cell = cc.per_bin_kernel_graphite(T_cell, a_cm, inv_m, Ye=1.0)
    Lam_dust_cell = float(np.dot(fracs_bins, K_cell))

    # Schure+09/DM72 gas cooling with metallicity scaling.
    Z   = float(getattr(cfg.cooling, "Z_OVER_ZSUN", 1.0))
    ion = int(getattr(cfg.cooling, "SCHURE_ION_DEG_INDEX", 0))

    # lambda_hd requires a one-dimensional hydrogen number-density array.
    X_H = 0.71
    m_H = 1.6735575e-24
    nH_scalar = float(cfg.cloud1.RHO) * X_H / m_H
    nH = np.full_like(T, nH_scalar, dtype=float)
    nH_cell = np.array([nH_scalar], dtype=float)                  # for single-point call

    sch_high, sch_low = _cooling_tables_paths()

    # Per-element weights: constant He (index 0), metals (1..14) scale with Z
    rel_weights = np.ones(15, float)
    rel_weights[1:] *= Z

    Lambda_gas = lambda_hd(T, nH, sch_high, sch_low, ionDegInd=ion, rel_weights=rel_weights)
    Lam_gas_cell = float(lambda_hd(np.array([T_cell], dtype=float), nH_cell, sch_high, sch_low,
                                   ionDegInd=ion, rel_weights=rel_weights)[0])

    # === Plot ==================================================================
    plt.figure(figsize=(8.4, 5.4))

    # Dust
    plt.plot(logT, np.log10(np.clip(Lambda_dust, 1e-300, np.inf)),
             lw=3, label=fr"Λ$_{{\rm dust}}$(T), $\mathcal{{D}}={D_plot:.0e}$")

    # Gas (Z-aware)
    plt.plot(logT, np.log10(np.clip(Lambda_gas, 1e-300, np.inf)),
             lw=2, ls="--", label=fr"Λ$_{{\rm gas}}$(T, Z={Z:g})")

    # Mark cloud1 point for both
    plt.scatter([np.log10(T_cell)], [np.log10(max(Lam_dust_cell, 1e-300))],
                s=50, marker="o", label="cloud1 @ T (dust)")
    plt.scatter([np.log10(T_cell)], [np.log10(max(Lam_gas_cell, 1e-300))],
                s=50, marker="x", label="cloud1 @ T (gas)")

    # Axis limits
    plt.xlim(4.0, 8.0)
    plt.ylim(-26.0, -19.0)

    plt.xlabel(r'$\log_{10}\,T\ \mathrm{[K]}$')
    plt.ylabel(r'$\log_{10}\ \Lambda\ \mathrm{[erg\,cm^{3}\,s^{-1}]}$')
    plt.grid(True, ls=':', alpha=0.35)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()

    # Save if --save provided OR if headless backend (Agg)
    should_save = bool(save_path) or "agg" in matplotlib.get_backend().lower()
    if should_save:
        out = Path(save_path) if save_path else (_outer / "images" / "cooling_curves.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out.with_suffix(".png"), dpi=160)
        print(f"[plot_cooling] saved: {out.with_suffix('.png')}")
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot dust and gas cooling curves and a sampled cloud1 point.")
    ap.add_argument("parfile", help="Path to tonantzintla.par")
    ap.add_argument("--D", type=float, default=None,
                    help="Override total dust-to-gas ratio used in the plot (e.g. 1e-2).")
    ap.add_argument("--save", type=str, default=None,
                    help="Write figure to this PNG path (no display).")
    args = ap.parse_args()
    main(args.parfile, args.D, args.save)
