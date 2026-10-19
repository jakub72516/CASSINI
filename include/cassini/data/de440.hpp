// Cassini — DE440 ephemeris wrapper over the SPK reader.
//
// Provides NAIF body IDs and heliocentric state queries in the simulator's
// natural units (AU, AU/day). DE440 stores positions in km and velocities in
// km/s referenced to the J2000 (ICRF) frame at barycenter ID 0.
#pragma once

#include <string>

#include "cassini/core/constants.hpp"
#include "cassini/core/vec3.hpp"
#include "cassini/data/spk.hpp"

namespace cassini::data {

// NAIF body IDs used by DE440.
enum class Body : int {
    SolarSystemBarycenter = 0,
    Sun = 10,
    MercuryBarycenter = 1,
    Mercury = 199,
    VenusBarycenter = 2,
    Venus = 299,
    EarthMoonBarycenter = 3,
    Earth = 399,
    Moon = 301,
    MarsBarycenter = 4,
    Mars = 499,
    JupiterBarycenter = 5,
    Jupiter = 599,
    SaturnBarycenter = 6,
    Saturn = 699,
    UranusBarycenter = 7,
    Uranus = 799,
    NeptuneBarycenter = 8,
    Neptune = 899,
};

constexpr double KM_PER_AU = constants::AU_m / 1000.0;       // 149597870.7 km
constexpr double KMS_PER_AUDAY = KM_PER_AU / constants::DAY_s;

class De440 {
public:
    bool load(const std::string& bsp_path) { return spk_.load_from_file(bsp_path); }

    bool available() const { return !spk_.segments().empty(); }

    // Heliocentric state (relative to the Sun) in AU and AU/day.
    // et = TDB ephemeris seconds past J2000.
    bool heliocentric_state(int body, double et, Vec3<double>& r, Vec3<double>& v) const {
        Vec3<double> r_km, v_kms;
        if (!spk_.state(body, static_cast<int>(Body::Sun), et, r_km, v_kms)) return false;
        r = r_km * (1.0 / KM_PER_AU);
        v = v_kms * (1.0 / KMS_PER_AUDAY);
        return true;
    }

    const SpkFile& spk() const { return spk_; }

private:
    SpkFile spk_;
};

}  // namespace cassini::data
