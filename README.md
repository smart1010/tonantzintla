# TONANTZINTLA

**TONANTZINTLA** is a three-dimensional finite-volume hydrodynamics code for controlled cloud–flow, shock, wind, and mixing calculations on a uniform Cartesian grid. The domain is decomposed into one-dimensional slabs along `x` for Message Passing Interface (MPI) parallelization.

## Development status and disclaimer

**TONANTZINTLA is scientific software under active development.** Its numerical methods, physical modules, documentation, and validation procedures may change. Three-dimensional hydrodynamic calculations were performed with the code. Validation remains limited across configurations, physical regimes, and computational environments.

Users should independently assess numerical convergence, conservation, and the applicability of the implemented physics before using results for scientific interpretation or publication. The software is provided **as is**, without any warranty of accuracy or fitness for a particular purpose.

## Authors and contributors

The following people developed or advised TONANTZINTLA. Citation metadata are provided in `CITATION.cff`.

- **Sergio Martínez González** (Instituto Nacional de Astrofísica, Óptica y Electrónica; INAOE): Author and lead developer.
- **Guillermo Tenorio-Tagle** (INAOE): Scientific advisor and feature design contributor.

## Dust physics and CINDER

The dust-physics implementation in **TONANTZINTLA** draws on **CINDER**, originally developed by **Sergio Martínez-González** for the Adaptive Mesh Refinement (AMR) **FLASH** code. The `tonantzintla/cinder/` package provides grain-size distributions, dust growth and sputtering, and dust-induced radiative cooling within TONANTZINTLA's hydrodynamic framework. The scientific development and applications of CINDER are described by Martínez-González et al. (2018, 2019, 2022).

The CLOUD-1000 example disables dust physics with `DUST_ON = False`; the dust treatment is an optional component of the code. Its numerical applicability and convergence should be assessed for each dust-enabled calculation.

## Requirements

- Python 3.8 or later
- NumPy
- SciPy
- Matplotlib
- h5py
- mpi4py for multi-process execution
- an MPI implementation compatible with the installed mpi4py

Serial execution runs independently of mpi4py. Multi-process execution requires mpi4py and a compatible MPI implementation. The program terminates when a multi-process launch lacks mpi4py.

## Cooling tables

The Schure cooling tables in `tonantzintla/cooling/tables/` and `tonantzintla/data/Schure_total.dat` are attributed to Schure et al. (2009). They provide temperature-dependent gas-cooling coefficients.

## Hydrodynamic method

The conserved state is

```text
U = [rho, rho*vx, rho*vy, rho*vz, E, eint]
```

where `E` is the total energy density and `eint` is the auxiliary internal-energy density used by the dual-energy treatment.

Available schemes are:

- `ORDER = 'first'`: piecewise-constant Godunov reconstruction with Forward Euler time integration.
- `ORDER = 'plm'`: primitive-variable piecewise linear method (PLM) reconstruction with a monotonized-central limiter and an unsplit MUSCL–Hancock half-step predictor.
- `RIEMANN = 'hlle'`: Harten–Lax–van Leer–Einfeldt (HLLE) approximate Riemann solver.
- `RIEMANN = 'hllc'`: Harten–Lax–van Leer–Contact (HLLC) solver with a local HLLE fallback for nonphysical star states.

The optional carbuncle treatment blends HLLC toward HLLE in compressive strong-shock regions. Its multidimensional sensor combines the normalized pressure jump, normal compression, and the immediate transverse H stencil. The sensor remains inactive for contacts and shear layers with negligible pressure jumps.

The MUSCL–Hancock reconstruction applies a local positivity fallback. Primitive slopes are flattened only where a predicted face state would have an anomalously small density or pressure relative to the half-time cell-centered state.

```ini
[solver]
ORDER = 'plm'
RIEMANN = 'hllc'
CARBUNCLE_FIX_ON = True
SHOCK_PRESSURE_THRESHOLD = 0.2
SHOCK_FULL_HLLE_THRESHOLD = 0.6
SHOCK_COMPRESSION_THRESHOLD = 0.05
CARBUNCLE_HLLE_BLEND = 1.0
HANCOCK_FLATTEN_FRACTION = 0.1
```

## MPI flux conservation

For every shared PLM interface, adjacent ranks use the owner-cell reconstructions from both sides and the same numerical face flux. The same convention is used for gas, passive tracers, and dust-bin transport. Internal MPI-face contributions therefore cancel discretely apart from floating-point reductions used by diagnostics.

## Cloud interfaces

Cloud boundaries may be sharp, use a fixed-width ramp, or select the width automatically from the density contrast.

```ini
[interfaces]
INTERFACE_RAMP_MODE = 'auto'
INTERFACE_RAMP_CELLS = 2
INTERFACE_RAMP_MIN_CELLS = 2
INTERFACE_RAMP_MAX_CELLS = 16
INTERFACE_RAMP_MAX_DEX_PER_CELL = 1.0
```

For the cosine transition used by the code, automatic mode estimates

```text
N_ramp = ceil[(pi/2) * abs(log10(rho_cloud/rho_ambient)) / max_dex_per_cell]
```

and clips the result to the configured limits and to the cloud geometry. Density is interpolated logarithmically. The quantity `sqrt(rho)*v` is interpolated with the same interface weight, which prevents the ramp from introducing kinetic-energy values above the convex combination of the endpoint states. Pressure is interpolated with the common weight and remains constant when the endpoint pressures are equal. Cloud tracers and initial dust fractions use the same ramp.

`INTERFACE_RAMP_MODE = 'sharp'` disables smoothing. `INTERFACE_RAMP_MODE = 'fixed'` uses `INTERFACE_RAMP_CELLS`.

## Runtime configuration

Each MPI rank parses `tonantzintla.par` independently. The generated settings source is compared across ranks and installed directly in memory before the numerical modules are imported. Rank 0 writes `tonantzintla/settings.py` as a provenance copy; configuration changes are made in `tonantzintla.par`.

The principal sections are:

- `[run]`: dimensionality, physics switches, integration time, Courant–Friedrichs–Lewy (CFL) number, output cadence, and restart controls.
- `[grid]`: resolution and Cartesian domain.
- `[ambient]`, `[cloud1]`, `[cloud2]`: thermodynamic and velocity states.
- `[turbulence]`: initial turbulent velocity amplitudes.
- `[cooling]`: gas cooling configuration.
- `[dust]`: dust distribution, growth, sputtering, and dust cooling.
- `[solver]`: reconstruction, Riemann solver, and shock stabilization.
- `[interfaces]`: cloud-interface regularization.
- `[wind]`: boundary inflow.
- `[frame]`: Galilean moving-frame transformation.
- `[plot]`: diagnostic plane, output directory, and PNG-family switches.

## CLOUD-1000 example

The root-level `tonantzintla.par` configures the CLOUD-1000 model described by
[Tenorio-Tagle & Martínez-González (2026, arXiv:2611.25049)](https://arxiv.org/abs/2611.25049): a 15 pc cloud at 500 K with an
initial density corresponding to approximately 10³ cm⁻³, moving at −1200 km/s
through a 5 × 10⁶ K intracluster medium with approximately 0.1 cm⁻³.
The domain uses `lref = 7`, a 512³ Cartesian mesh, PLM reconstruction,
the HLLE Riemann solver, passive tracers, and an adiabatic moving frame.

This example starts from initial conditions using `RESTART = False`,
`RESTART_SNAP = 0`, and `SNAP_NEXT_OVERRIDE = 0`. The restart-related
snapshot numbers are inactive with `RESTART = False`. Simulation
snapshots are written every 10,000 years according to `SNAP_EVERY`. A copy of
this configuration is available as `examples/tonantzintla.par`.

## Execution

The following commands cover serial, MPI, and Slurm execution. Distributed runs require a compatible MPI stack for the Python executable, mpi4py, and the launcher.

Serial:

```bash
python -u -m tonantzintla.main tonantzintla.par
```

MPI:

```bash
mpiexec -np 24 python -u -m tonantzintla.main tonantzintla.par
```

Slurm:

```bash
srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" \
  python -u -m tonantzintla.main tonantzintla.par
```

## Output and restart data

Rank 0 coordinates HDF5 snapshot and checkpoint writes, processing one slab at a time. PNG and ASCII diagnostics gather only the selected central plane. Checkpoints contain conserved fields, the auxiliary internal energy, tracers, dust bins, time-step metadata, snapshot counters, and moving-frame metadata.

The snapshot scheduler is based on simulation time. On restart, its phase is initialized from the checkpoint time. Each output records the actual evolved time of its state. The source archive contains code, configuration files, and cooling tables.

## Diagnostic plots

The standard slice family contains density, temperature, dust-to-gas ratio when enabled, and velocity magnitude. The energy diagnostic contains internal-energy density, kinetic-energy density, the signed vorticity component normal to the selected plane, and the internal-to-total energy ratio.

For an `xz` cut,

```text
omega_y = d(vx)/dz - d(vz)/dx.
```

The two PNG families are controlled independently:

```ini
[plot]
SNAP_PLOTS_ON = True
DIAG_PLOTS_ON = True
```

Repeated INI sections are accepted; later entries override earlier entries for repeated keys. These switches affect only `snap_*.png` and `diag_energy_*.png`. ASCII slices and HDF5 outputs remain enabled.

Color limits are determined only from the initial central slice and stored in

```text
plot_limits_<BASEN>.json
```

All subsequent PNGs use the initial color limits, including after restarts. Logarithmic limits are expanded outward to the nearest 0.5 dex. Linear velocity and vorticity limits retain their initial-slice ranges. A restart with PNG plots enabled requires the original limits file and stops with an explicit error when that file is missing.

## Recommended verification

- uniform-flow preservation;
- Sod and Lax shock tubes;
- stationary and advected contacts;
- a strong grid-aligned shock with an odd–even perturbation;
- a blunt-body bow shock;
- identical calculations with different MPI slab counts;
- global conservation before cooling, dust source terms, and physical boundary fluxes;
- cloud-interface tests over the intended density-contrast range;
- restart tests confirming output cadence and fixed color limits.

For a shared MPI interface, the numerical flux on the adjacent ranks should be bitwise identical. Runs with different process counts should agree within floating-point and reduction-order tolerances.

## References

- B. van Leer, 1979, *Journal of Computational Physics*, 32, 101–136, doi:10.1016/0021-9991(79)90145-1.
- J. J. Quirk, 1994, *International Journal for Numerical Methods in Fluids*, 18, 555–574, doi:10.1002/fld.1650180603.
- R. Sanders, E. Morano, and M.-C. Druguet, 1998, *Journal of Computational Physics*, 145, 511–537, doi:10.1006/jcph.1998.6047.
- J. M. Stone et al., 2008, *The Astrophysical Journal Supplement Series*, 178, 137–177, doi:10.1086/588755.
- K. M. Schure, D. Kosenko, J. S. Kaastra, R. Keppens, and J. Vink, 2009, *Astronomy & Astrophysics*, **508**, 751–757, [doi:10.1051/0004-6361/200912495](https://doi.org/10.1051/0004-6361/200912495). *A new radiative cooling curve based on an up-to-date plasma emission code*.
- Martínez-González, S., et al., 2018, *The Astrophysical Journal*, **866**, 40, [doi:10.3847/1538-4357/aadb88](https://doi.org/10.3847/1538-4357/aadb88). *Pyroclastic Blowout: Dust Survival in Isolated versus Clustered Supernovae*.
- Martínez-González, S., et al., 2019, *The Astrophysical Journal*, **887**, 198, [doi:10.3847/1538-4357/ab571b](https://doi.org/10.3847/1538-4357/ab571b). *Supernovae within Pre-existing Wind-blown Bubbles: Dust Injection versus Ambient Dust Destruction*.
- Martínez-González, S., et al., 2022, *The Astrophysical Journal*, **934**, 51, [doi:10.3847/1538-4357/ac77fe](https://doi.org/10.3847/1538-4357/ac77fe). *Dust Grain Growth and Dusty Supernovae in Low-metallicity Molecular Clouds*.
