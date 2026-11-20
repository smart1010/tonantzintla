"""Plot gas and graphite-dust cooling functions for a TONANTZINTLA configuration."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from tonantzintla.read_data import load_config, write_settings_module


def load_schure_lambda(table_path):
    data = np.loadtxt(table_path, comments="#")
    log_t, log_lambda = data[:, 0], data[:, 1]

    def schure_lambda(temperature):
        values = np.log10(np.clip(temperature, 1e-2, None))
        interpolated = np.interp(values, log_t, log_lambda,
                                     left=log_lambda[0], right=log_lambda[-1])
        return 10.0 ** interpolated

    return schure_lambda


def main(parfile="tonantzintla.par"):
    cfg = load_config(parfile)
    settings_path = Path(__file__).resolve().parents[1] / "settings.py"
    write_settings_module(cfg, settings_path)

    from tonantzintla import settings as S
    from tonantzintla.cinder import cinder_massfrac as cm
    from tonantzintla.cinder import cinder_cool as cc

    temperature = np.logspace(1, 9, 500)
    lambda_gas = load_schure_lambda(S.LAMBDA_TABLE_FILE)(temperature)

    edges_um, mass_fracs = cm.make_bins()
    radii_cm, inv_m = cm.representative_sizes(edges_um)
    dust_ratio = 1.0e-2
    kernels = cc.per_bin_kernel_graphite(temperature, radii_cm, inv_m)
    lambda_dust = dust_ratio * np.dot(mass_fracs, kernels)

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(temperature, lambda_gas, label="Schure (gas) cooling")
    plt.loglog(temperature, lambda_dust, "--", label="Dust cooling (graphite)")
    plt.xlabel("Temperature [K]")
    plt.ylabel(r"$\Lambda\ \mathrm{[erg\ cm^{3}\ s^{-1}]}$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tonantzintla.par")
