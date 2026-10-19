// Cassini — HDF5 export of particle snapshots (optional feature, gated behind
// CASSINI_ENABLE_HDF5). Writes a dataset of [time, id, mass, x, y, z, vx, vy,
// vz] per body using the HDF5 C API.
#include <cstdint>
#include <string>
#include <vector>

#include <hdf5.h>

#include "cassini/physics/system.hpp"

namespace cassini::data {

// Export the current system state to an HDF5 file (appends to a growable
// dataset keyed by time). Returns true on success.
inline bool export_hdf5(const cassini::System& sys, const std::string& path) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return false;

    const std::size_t n = sys.size();
    std::vector<double> buffer;
    buffer.reserve(n * 9);
    for (std::size_t i = 0; i < n; ++i) {
        buffer.push_back(sys.t);
        buffer.push_back(static_cast<double>(i));
        buffer.push_back(sys.mass[i]);
        buffer.push_back(sys.pos[i].x);
        buffer.push_back(sys.pos[i].y);
        buffer.push_back(sys.pos[i].z);
        buffer.push_back(sys.vel[i].x);
        buffer.push_back(sys.vel[i].y);
        buffer.push_back(sys.vel[i].z);
    }

    const hsize_t dims[2] = {static_cast<hsize_t>(n), 9};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t dset = H5Dcreate2(file, "/snapshot", H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT,
                            H5P_DEFAULT);
    const herr_t status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                   buffer.data());
    H5Dclose(dset);
    H5Sclose(space);
    H5Fclose(file);
    return status >= 0;
}

}  // namespace cassini::data
