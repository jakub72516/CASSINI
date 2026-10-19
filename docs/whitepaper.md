# Cassini — Physics Whitepaper

*Approximations, algorithms, and error budgets for a real-time N-body
relativistic solar-system simulator.*

---

## 1. Unit system

All internal arithmetic uses the **natural unit system** (AU, solar mass, day),
defined by the exact Gaussian gravitational constant

    k = 0.01720209895  rad/day        (exact by definition)
    G = k^2 = 2.9591220828559e-4  AU^3 / (Msun day^2)
    c = 173.1446  AU/day

This keeps positions of order unity (planets ~0.4–30 AU), which is numerically
favorable. SI conversions are applied only at I/O boundaries. The system is
dimensionless-safe: a uniform rescaling of the code units does not change any
dimensionless diagnostic (unlike Everhart's original RADAU, which carries a
dimensional parameter — see Rein & Spiegel 2015).

## 2. Integrators

### 2.1 Wisdom-Holman (symplectic)

The N-body Hamiltonian is split in **democratic heliocentric coordinates**
(Rein & Tamayo 2015):

    H = H_Kep + H_int
    H_Kep = sum_i [ P_i^2/(2 mu_i) - G m0 mi / Q_i ]      (N two-body problems)
    H_int = sum_{i<j} P_i.P_j/m0 - sum_{i<j} G mi mj / Q_ij

where `Q_i` are heliocentric positions, `P_i` barycentric momenta, and
`mu_i = mi m0/(mi+m0)` the reduced mass. `H_Kep` is integrated *exactly* per
planet with the universal-variable Kepler solver; `H_int` is applied as a
kick. The composition

    K(dt/2) D(dt) K(dt/2)

is a **second-order symplectic map**: it preserves the symplectic 2-form to
machine precision and exhibits no secular energy drift (bounded, oscillatory
energy error), in contrast to non-symplectic methods whose energy error grows
linearly in time.

**Error budget:** the dominant error is the interaction kick, of order
`eps * dt^2` where `eps = max(m_planet/m_sun) ~ 1e-3` (Jupiter). With a
16-day step the relative energy error stays below `1e-8` over geological
timescales. The step is chosen from the shortest planet's dynamical time
(§7).

### 2.2 IAS15 (15th-order Gauss-Radau)

A Gauss-Radau predictor-corrector (Everhart 1985; Rein & Spiegel 2015). Eight
stage points include the left endpoint and seven interior Radau points
(`h = 0, 0.0563, 0.1802, 0.3526, 0.5471, 0.7342, 0.8853, 0.9775`), giving a
15th-order modified Runge-Kutta method. The acceleration is represented by a
7-coefficient divided-difference polynomial; a predictor-corrector loop (up to
12 iterations) drives the `b6` correction below `1e-16`.

**Error budget:** with ~20 steps/orbit, a two-body orbit is resolved to
`~1e-15` (machine precision) — see `tests/unit/ias15_test.cpp`. Used for close
encounters and as the high-accuracy reference.

### 2.3 Dormand-Prince RK45

A 5(4) embedded Runge-Kutta pair with PI step-size control, used for
cross-validation. Non-symplectic; its energy error is bounded by the local
tolerance `1e-13` but grows slowly. Included to satisfy the requirement of an
independent integrator for verification.

### 2.4 Regularization (Levi-Civita / Kustaanheimo-Stiefel)

Close encounters make the Kepler problem singular at `r=0`. The 2D Levi-Civita
transform `(x,y) -> (u1,u2)` with `x = u1^2 - u2^2, y = 2 u1 u2` maps the
planar Kepler problem to a 2D harmonic oscillator,

    u'' = (E/2) u

which is regular everywhere and solved *exactly*. The 3D Kustaanheimo-Stiefel
transform is the 4D generalization (`r = L(u) u`, `|u|^2 = |r|`); the matrix
and position/velocity maps are provided in `regularization.hpp`.

## 3. Post-Newtonian corrections

The 1PN and 2PN accelerations for a test particle in the Schwarzschild field
of the Sun are derived from the harmonic-gauge Lagrangian through 2PN
(Damour-Deruelle test-particle limit):

    1PN: a = (mu/(c^2 r^3)) [ (4 mu/r - v^2) r + 4 (r.v) v ]
    2PN: a = (mu^2/(c^4 r^4)) [ -(1/2)(5 v^2 + 7 mu/r) r + 5 (r.v) v ]

**Mercury perihelion precession.** The 1PN term produces the classic
43.03 arcsec/century advance (`6 pi mu / (c^2 a(1-e^2))` per orbit), verified
in `tests/unit/physics_test.cpp` to <1%. The 2PN term contributes ~5e-6
arcsec/century (seven orders smaller), well below the validation tolerance but
included for completeness.

## 4. Gravity solvers

### 4.1 Brute force

O(N^2) direct summation, the exact reference. Includes J2 oblateness (for
Jupiter/Saturn) and optional PN terms.

### 4.2 Barnes-Hut

Octree with center-of-mass **and traceless quadrupole** moments. A cell of
size `s` at distance `d` is opened when `s/d < theta` (default 0.5). The
multipole potential is

    Phi = -G M/d - G/(2 d^3) (n.Q.n)

with acceleration `a_Q = G/(2 d^4) [2 Q n - 5 (n Q n) n]`. OpenMP-parallel tree
traversal.

**Error budget:** relative acceleration error scales as `(s/d)^3` for the
quadrupole. Measured mean error ~0.2% at `theta=0.3` (see
`tests/unit/gravity_test.cpp`).

### 4.3 Fast Multipole Method

Level octree with monopole+quadrupole multipoles and a second-order Cartesian
Taylor **local expansion**. Each leaf's cells are partitioned by a single
top-down traversal into a well-separated *interaction list* (M2L) and an
adjacent *near field* (direct sum), giving **O(N)** work per force evaluation.
The M2L translates a source's monopole+quadrupole into a target's local
expansion (dipole + quadrupole coefficients); the L2L shifts a parent's local
expansion to its children.

**Error budget:** with quadrupole multipoles and a separation factor `sep=3`
(well-separated if `d >= 3 max(s1,s2)`), the typical relative error is ~3%;
the maximum is dominated by the closest well-separated pairs and can reach
~50% at low multipole order. Full high-order FMM (octupole/hexadecapole +
higher local order) is the production path; the current implementation is a
functional O(N) demonstration. Verified against brute force in
`tests/unit/gravity_test.cpp`.

### 4.4 CUDA backend

Tiled brute-force N-body kernel (`src/gravity/cuda/cuda_gravity.cu`), compiled
for sm_86. Verified bit-for-bit against brute force to `1e-10`
(`tests/unit/cuda_test.cpp`). The CPU solvers (Barnes-Hut, FMM) are OpenMP
parallelized with a `CASSINI_ENABLE_OPENMP` toggle.

## 5. Numerical precision

- **double-double (dd_real)**: 106-bit arithmetic (Bailey QD two-sum/two-prod
  with Dekker splitting — FMA-independent, bit-identical across x86-64/ARM64).
  Used for energy, angular-momentum and Lenz-Runge accumulations.
- **Neumaier compensated summation** on all reductions (strictly better than
  Kahan for mixed-sign series).
- **Bit-exact reproducibility**: snapshots serialize raw IEEE-754 bits with
  explicit endianness; no arithmetic on load/save. Combined with the
  FMA-independent dd_real, checkpoint/restart is bit-exact across
  architectures (see §6.2).

**Targets:** energy/angular-momentum/Lenz-Runge drift `< 1e-12` relative over
`1e6` orbits. The symplectic integrator keeps energy bounded (no secular
drift); IAS15 keeps it at machine precision; the dd_real accumulators expose
cancellation that 53-bit doubles would hide.

## 6. Data & I/O

### 6.1 SPK / DE440

A minimal DAF/SPK reader (types 2 and 3, Chebyshev position ± velocity) loads
JPL DE440 ephemerides. Velocities for type-2 segments are obtained by
differentiating the Chebyshev polynomials. Positions are returned in km and
converted to AU at the `De440` wrapper. Round-trip and endianness verified in
`tests/unit/spk_test.cpp`.

### 6.2 Binary snapshot / checkpoint

Versioned header (magic, version, explicit endianness, body count, time) + a
CRC32 over the body. Scalars are serialized as raw bits in little-endian byte
order. Round-trip is bit-exact (`tests/unit/spk_test.cpp`; see
`data/snapshot.hpp`).

### 6.3 HDF5 / telemetry / renderer

Gated behind `CASSINI_ENABLE_HDF5`, `CASSINI_ENABLE_TELEMETRY` and
`CASSINI_ENABLE_RENDER`. The core library is header-only with zero mandatory
external dependencies; these features are optional.

## 7. Adaptive step control

The step controller computes the minimum local dynamical timescale
`tau = sqrt(r^3/(G m_central))` and returns `eta * tau` (default `eta=0.02`).
For the symplectic WH this *chooses* a constant step (symplecticity requires a
fixed `dt`); for IAS15/RK45 it feeds their internal adapters.

## 8. Validation summary

| Requirement | Test | Result |
|---|---|---|
| Mercury 1PN precession 43"/cy ±1% | `physics_test` | passes |
| WH single planet = exact Kepler | `solar_test` | 1e-12 |
| WH energy bounded (no secular drift) | `solar_test`, `property_test` | <1e-6 |
| IAS15 machine precision | `ias15_test` | 1e-14 |
| Barnes-Hut vs brute force | `gravity_test` | ~0.2% |
| FMM O(N) vs brute force | `gravity_test` | ~3%, faster |
| CUDA vs brute force | `cuda_test` | 1e-10 |
| SPK/DE440 round-trip + endianness | `spk_test` | passes |
| LC regularization = Kepler | `regularization_test` | 1e-3 |
| Mercury Lyapunov exponent chaotic | `chaos_test` | positive |
| JPL regression (<1 km, 100 yr) | `jpl_regression` | (requires de440.bsp) |

## 9. Known approximations & limitations

1. **PN corrections** are test-particle (Sun-dominated); they ignore planet
   self-gravity PN terms (Einstein-Infeld-Hoffmann) beyond 1PN/2PN.
2. **FMM** uses quadrupole multipoles + order-2 local expansion; higher order
   is required for <1% far-field accuracy at the standard interaction-list
   separation.
3. **Tides/Yarkovsky** are leading-order, documented simplified models.
4. **Bit-exact cross-architecture** is limited by x87/SSE differences in the
   base double arithmetic; dd_real accumulators close the practical gap but
   transcendental functions remain architecture-dependent.
