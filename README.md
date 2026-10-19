# Cassini

A real-time N-body **relativistic** solar-system simulator written in modern C++20,
with an adaptive-mesh gravity solver, post-Newtonian corrections, and bit-exact
checkpointing — validated against JPL DE440 ephemerides.

> This is an ambitious, research-grade codebase. See `docs/whitepaper.md` for the
> physics approximations and their error budgets.

## Features

- **Integrators**: Wisdom-Holman symplectic map, IAS15 (15th-order Gauss-Radau),
  Dormand-Prince RK45 (cross-validation), Kustaanheimo-Stiefel regularization.
- **Relativity**: 1PN + 2PN post-Newtonian terms reproducing Mercury's 43"/cy precession.
- **Gravity solvers**: brute force, Barnes-Hut octree (quadrupole), Fast Multipole
  Method (O(N)), with OpenMP + SIMD and optional CUDA backends.
- **Precision**: double-double (128-bit) accumulators, Neumaier compensated summation,
  energy/angular-momentum/Lenz-Runge drift tracking.
- **Data**: versioned binary snapshots, HDF5/SPICE (SPK) export, DE440 reader,
  bit-exact checkpoint/restart.
- **Tests**: property-based (rapidcheck), JPL Horizons regression (<1 km inner planets),
  chaos/Lyapunov validation.

## Build

```sh
# Configure + build (tests + benchmarks enabled)
cmake --preset dev
cmake --build --preset dev

# Run tests
ctest --preset dev

# CUDA backend (requires CUDA toolkit on PATH or CUDAToolkit_ROOT)
cmake --preset cuda
cmake --build --preset cuda
```

Dependencies are managed with [vcpkg](https://vcpkg.io) (`vcpkg.json`); the core
numerics/physics library is header-only with zero external dependencies, so a bare
build works without vcpkg. HDF5, gRPC and the renderer are optional feature toggles.

## Layout

```
include/cassini/   public headers (core, math, physics, gravity, data, analysis)
src/               app, CUDA kernels, HDF5 glue
tests/             unit, property, regression, chaos
benchmarks/        throughput + scaling
docs/              Doxygen + physics whitepaper
```

## License

See `LICENSE`.
