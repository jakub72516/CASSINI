// Cassini — CUDA gravity kernel (tiled brute-force N-body). Compiled with
// nvcc; the host-side ForceModel wrapper is in cuda_force.hpp.
#include <cuda_runtime.h>

namespace {

__global__ void gravity_kernel(const double* __restrict__ pos,
                               const double* __restrict__ mass,
                               double* __restrict__ acc,
                               int n,
                               double G) {
    extern __shared__ double sh[];  // tiled positions (3 * blockDim.x doubles)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = pos[3 * i];
        yi = pos[3 * i + 1];
        zi = pos[3 * i + 2];
    }
    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int t = 0; t < gridDim.x; ++t) {
        const int j = t * blockDim.x + tid;
        if (j < n) {
            sh[3 * tid] = pos[3 * j];
            sh[3 * tid + 1] = pos[3 * j + 1];
            sh[3 * tid + 2] = pos[3 * j + 2];
        }
        __syncthreads();
        for (int k = 0; k < blockDim.x; ++k) {
            const int j2 = t * blockDim.x + k;
            if (j2 >= n || j2 == i) continue;
            const double dx = sh[3 * k] - xi;
            const double dy = sh[3 * k + 1] - yi;
            const double dz = sh[3 * k + 2] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r = sqrt(r2);
            if (r == 0.0) continue;
            const double inv = G * mass[j2] / (r2 * r);
            ax += inv * dx;
            ay += inv * dy;
            az += inv * dz;
        }
        __syncthreads();
    }

    if (i < n) {
        acc[3 * i] = ax;
        acc[3 * i + 1] = ay;
        acc[3 * i + 2] = az;
    }
}

}  // namespace

extern "C" void cassini_cuda_gravity(const double* pos, const double* mass, double* acc, int n,
                                     double G) {
    double* d_pos = nullptr;
    double* d_mass = nullptr;
    double* d_acc = nullptr;
    const std::size_t nbytes = static_cast<std::size_t>(n) * 3 * sizeof(double);
    cudaMalloc(&d_pos, nbytes);
    cudaMalloc(&d_mass, static_cast<std::size_t>(n) * sizeof(double));
    cudaMalloc(&d_acc, nbytes);
    cudaMemcpy(d_pos, pos, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, static_cast<std::size_t>(n) * sizeof(double), cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = (n + block - 1) / block;
    gravity_kernel<<<grid, block, static_cast<std::size_t>(block) * 3 * sizeof(double)>>>(
        d_pos, d_mass, d_acc, n, G);

    cudaMemcpy(acc, d_acc, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_pos);
    cudaFree(d_mass);
    cudaFree(d_acc);
}
