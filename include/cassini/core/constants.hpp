// Cassini — physical constants in the natural unit system (AU, solar mass, day)
// plus SI reference values for input/output conversion.
//
// The Gaussian gravitational constant k defines the unit system exactly:
//      k = 0.01720209895  rad / day
//      GM_sun = k^2       AU^3 / day^2        (with M_sun = 1)
#pragma once

#include <cmath>

namespace cassini::constants {

// --- Natural units (AU, Msun, day) ---
inline constexpr double k_gaussian = 0.01720209895;  // exact by definition
inline constexpr double G = k_gaussian * k_gaussian; // AU^3 / (Msun day^2)
inline constexpr double k2 = G;

// --- SI references ---
inline constexpr double AU_m = 1.495978707e11;          // astronomical unit [m]
inline constexpr double MSUN_kg = 1.98847e30;           // solar mass [kg]
inline constexpr double DAY_s = 86400.0;                // day [s]
inline constexpr double JULIAN_YEAR_DAYS = 365.25;      // [day]
inline constexpr double C_SI = 299792458.0;             // speed of light [m/s]

// --- Speed of light in natural units [AU/day] ---
inline constexpr double C = C_SI * DAY_s / AU_m;        // ~173.1446 AU/day

// --- Solar luminosity in natural units [Msun AU^2 / day^3] ---
inline constexpr double L_SUN_SI = 3.828e26;            // W
inline constexpr double L_SUN = L_SUN_SI * DAY_s * DAY_s * DAY_s /
                                (MSUN_kg * AU_m * AU_m);

// --- Conversion helpers ---
inline constexpr double meters_to_au(double m) { return m / AU_m; }
inline constexpr double au_to_meters(double au) { return au * AU_m; }
inline constexpr double seconds_to_days(double s) { return s / DAY_s; }
inline constexpr double days_to_seconds(double d) { return d * DAY_s; }
inline constexpr double kg_to_msun(double kg) { return kg / MSUN_kg; }
inline constexpr double msun_to_kg(double m) { return m * MSUN_kg; }

}  // namespace cassini::constants
