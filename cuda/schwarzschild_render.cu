// Phase 1.5 — standalone CUDA Schwarzschild ray tracer.
//
// Mirrors the Rust integrator (analytic Christoffels, fixed-step RK4) so
// the GPU output is *bit-comparable* to the CPU output up to fp64
// associativity.  This is the minimum demonstration that the same physics
// runs on a B200 SM, one ray per thread.
//
// Build:   nvcc -O3 -arch=sm_100 schwarzschild_render.cu -o sch_gpu
//          (sm_100 = Blackwell; nvcc 13.x supports it.)
//
// Run:     ./sch_gpu                       # writes outputs/sch_gpu.ppm
//          ./sch_gpu 1024 1024 25 50 0.04
//
// Output is a P6 binary PPM so we don't need any imaging libs on the GPU
// host.  Convert to PNG with `convert sch_gpu.ppm sch_gpu.png` if desired.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define M_PI_F 3.141592653589793

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

// ---- physics ----------------------------------------------------------- //

// y = [t, r, θ, φ, p^t, p^r, p^θ, p^φ]
__device__ inline void rhs_schwarzschild(double M, const double* y, double* dy) {
    const double r     = y[1];
    const double theta = y[2];
    const double pt    = y[4];
    const double pr    = y[5];
    const double pth   = y[6];
    const double pph   = y[7];

    const double f     = 1.0 - 2.0 * M / r;
    const double sin_t = sin(theta);
    const double cos_t = cos(theta);

    dy[0] = pt;
    dy[1] = pr;
    dy[2] = pth;
    dy[3] = pph;

    const double gtr = M / (r * r * f);                 // Γ^t_{tr}
    const double grr = -M / (r * r * f);                // Γ^r_{rr}
    const double grtt = M * f / (r * r);                // Γ^r_{tt}
    const double grth = -(r - 2.0 * M);                 // Γ^r_{θθ}
    const double grph = -(r - 2.0 * M) * sin_t * sin_t; // Γ^r_{φφ}
    const double gthr = 1.0 / r;                        // Γ^θ_{rθ}
    const double gthph = -sin_t * cos_t;                // Γ^θ_{φφ}
    const double gphr = 1.0 / r;                        // Γ^φ_{rφ}
    const double cot_safe = (fabs(sin_t) < 1e-8) ? 0.0 : cos_t / sin_t;

    // dp^t/dλ = -Γ^t_{αβ} p^α p^β = -2 Γ^t_{tr} p^t p^r
    dy[4] = -2.0 * gtr * pt * pr;
    // dp^r/dλ
    dy[5] = -(
        grtt * pt * pt
      + grr  * pr * pr
      + grth * pth * pth
      + grph * pph * pph
    );
    // dp^θ/dλ = -(2 Γ^θ_{rθ} p^r p^θ + Γ^θ_{φφ} p^φ p^φ)
    dy[6] = -(2.0 * gthr * pr * pth + gthph * pph * pph);
    // dp^φ/dλ = -(2 Γ^φ_{rφ} p^r p^φ + 2 cot θ p^θ p^φ)
    dy[7] = -(2.0 * gphr * pr * pph + 2.0 * cot_safe * pth * pph);
}

__device__ inline void rk4_step(double M, double h, const double* y, double* y_next) {
    double k1[8], k2[8], k3[8], k4[8], yt[8];
    rhs_schwarzschild(M, y, k1);
    for (int i = 0; i < 8; ++i) yt[i] = y[i] + 0.5 * h * k1[i];
    rhs_schwarzschild(M, yt, k2);
    for (int i = 0; i < 8; ++i) yt[i] = y[i] + 0.5 * h * k2[i];
    rhs_schwarzschild(M, yt, k3);
    for (int i = 0; i < 8; ++i) yt[i] = y[i] + h * k3[i];
    rhs_schwarzschild(M, yt, k4);
    const double sixth = h / 6.0;
    for (int i = 0; i < 8; ++i)
        y_next[i] = y[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

// ---- kernel ------------------------------------------------------------ //

// Termination codes: 0=diverged, 1=captured, 2=escaped.
__global__ void trace_kernel(
    int width, int height,
    double r_obs, double fov_y, double M,
    double step, int max_steps,
    double horizon_pad, double escape_radius_M,
    int* term, double* final_state
) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    const int idx = py * width + px;

    // Build the local tetrad analytically for Schwarzschild at observer
    // (t=0, r=r_obs, θ=π/2, φ=0). Spatial axes: 1 = radial, 2 = θ, 3 = φ.
    const double theta_obs = 1.5707963267948966;
    const double f = 1.0 - 2.0 * M / r_obs;
    const double sf = sqrt(f);
    const double e0_t  = 1.0 / sf;
    const double e1_r  = sf;
    const double e2_th = 1.0 / r_obs;
    const double e3_ph = 1.0 / (r_obs * sin(theta_obs));

    // Pixel → local direction (forward = -e1, right = +e3, up = -e2).
    const double aspect = (double)width / (double)height;
    const double h_im = tan(0.5 * fov_y);
    const double w_im = h_im * aspect;
    const double u = (2.0 * (px + 0.5) / width  - 1.0) * w_im;
    const double v = -(2.0 * (py + 0.5) / height - 1.0) * h_im;

    const double dx_l = -1.0;
    const double dy_l = -v;
    const double dz_l = u;
    const double n_local = sqrt(dx_l * dx_l + dy_l * dy_l + dz_l * dz_l);
    const double n1 = dx_l / n_local;
    const double n2 = dy_l / n_local;
    const double n3 = dz_l / n_local;

    double y[8];
    y[0] = 0.0;
    y[1] = r_obs;
    y[2] = theta_obs;
    y[3] = 0.0;
    y[4] = e0_t;                        // p^t
    y[5] = n1 * e1_r;                   // p^r
    y[6] = n2 * e2_th;                  // p^θ
    y[7] = n3 * e3_ph;                  // p^φ

    int t = 0;  // diverged
    for (int n = 0; n < max_steps; ++n) {
        const double r = y[1];
        if (!isfinite(r)) { t = 0; break; }
        if (r < horizon_pad * 2.0 * M)         { t = 1; break; }
        if (r > escape_radius_M * M)           { t = 2; break; }

        double y_next[8];
        rk4_step(M, step, y, y_next);
        // Quick NaN guard.
        if (!isfinite(y_next[1])) { t = 0; break; }
        for (int i = 0; i < 8; ++i) y[i] = y_next[i];
    }

    term[idx] = t;
    for (int i = 0; i < 8; ++i) final_state[idx * 8 + i] = y[i];
}

// ---- host sky sampling (mirrors Rust sky.rs roughly) ------------------- //

static unsigned long long hash1(unsigned long long x) {
    x = x * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30; x = x * 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x = x * 0x94D049BB133111EBULL;
    x ^= x >> 31; return x;
}

static void sky_sample(const double dir[3], unsigned char rgb[3]) {
    const double x = dir[0], y = dir[1], z = dir[2];
    const double r = sqrt(x * x + y * y + z * z);
    const double theta = acos(z / fmax(r, 1e-30));
    double phi = atan2(y, x);
    if (phi < 0) phi += 2.0 * M_PI_F;

    const double phi_s = fmod(phi + 0.37, 2.0 * M_PI_F);
    const double th_s  = fmin(fmax(theta + 0.21, 0.0), M_PI_F);
    const int sec = (int)(phi_s / (2.0 * M_PI_F) * 6.0) % 6;

    static const double palette[6][3] = {
        {0.15, 0.08, 0.20}, {0.08, 0.15, 0.22}, {0.20, 0.15, 0.08},
        {0.10, 0.18, 0.10}, {0.22, 0.10, 0.10}, {0.10, 0.10, 0.22},
    };
    const int band = (int)(th_s / M_PI_F * 3.0);
    const double band_mod = (band % 2 == 0) ? 1.0 : 0.6;

    const int ic = (int)(phi_s / (2.0 * M_PI_F) * 24.0);
    const int jc = (int)(th_s  / M_PI_F * 12.0);
    const double check = ((ic + jc) % 2 == 0) ? 1.0 : 0.55;

    double col[3] = {
        palette[sec][0] * band_mod * check,
        palette[sec][1] * band_mod * check,
        palette[sec][2] * band_mod * check,
    };

    // Stars.
    const unsigned long long h = hash1(((unsigned long long)((phi_s / (2.0 * M_PI_F)) * 512.0)) * 73856093ULL
        ^ ((unsigned long long)((th_s / M_PI_F) * 256.0)) * 19349663ULL);
    if ((h & 0xFFFF) / 65535.0 < 0.004) {
        const double br = 0.4 + 0.6 * (((h >> 16) & 0xFF) / 255.0);
        const double te = ((h >> 24) & 0xFF) / 255.0;
        col[0] = fmin(col[0] + br * (1.0 - 0.3 * te), 1.0);
        col[1] = fmin(col[1] + br * (1.0 - 0.1 * fabs(te - 0.5)), 1.0);
        col[2] = fmin(col[2] + br * (0.7 + 0.3 * te), 1.0);
    }

    const double inv_g = 1.0 / 2.2;
    for (int c = 0; c < 3; ++c)
        rgb[c] = (unsigned char)(fmin(fmax(pow(col[c], inv_g), 0.0), 1.0) * 255.0);
}

static void asymptotic_dir(const double* y, double dir[3]) {
    const double r = y[1], theta = y[2], phi = y[3];
    const double pr = y[5], pth = y[6], pph = y[7];
    const double s = sin(theta), c = cos(theta), sp = sin(phi), cp = cos(phi);
    dir[0] = s * cp * pr + r * c * cp * pth - r * s * sp * pph;
    dir[1] = s * sp * pr + r * c * sp * pth + r * s * cp * pph;
    dir[2] = c      * pr - r * s     * pth;
    const double n = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (n > 0) { dir[0] /= n; dir[1] /= n; dir[2] /= n; }
}

// ---- main ------------------------------------------------------------- //

int main(int argc, char** argv) {
    int width = 1024, height = 1024;
    double r_obs = 25.0, fov_deg = 50.0, step = 0.04;
    int max_steps = 50000;
    if (argc > 1) width    = std::atoi(argv[1]);
    if (argc > 2) height   = std::atoi(argv[2]);
    if (argc > 3) r_obs    = std::atof(argv[3]);
    if (argc > 4) fov_deg  = std::atof(argv[4]);
    if (argc > 5) step     = std::atof(argv[5]);
    if (argc > 6) max_steps = std::atoi(argv[6]);

    const int n_pixels = width * height;
    int* d_term;
    double* d_final;
    CUDA_CHECK(cudaMalloc(&d_term,  n_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final, n_pixels * 8 * sizeof(double)));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    trace_kernel<<<grid, block>>>(
        width, height, r_obs, fov_deg * 3.141592653589793 / 180.0, 1.0,
        step, max_steps, 1.001, 1000.0,
        d_term, d_final);
    cudaEventRecord(b);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);

    int* term = new int[n_pixels];
    double* final_state = new double[n_pixels * 8];
    CUDA_CHECK(cudaMemcpy(term,        d_term,  n_pixels * sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_state, d_final, n_pixels * 8 * sizeof(double), cudaMemcpyDeviceToHost));

    int n_cap = 0, n_esc = 0, n_div = 0;
    unsigned char* img = new unsigned char[n_pixels * 3];
    for (int i = 0; i < n_pixels; ++i) {
        if (term[i] == 1)      { img[i*3]=0; img[i*3+1]=0; img[i*3+2]=0; ++n_cap; }
        else if (term[i] == 2) {
            double dir[3]; asymptotic_dir(&final_state[i*8], dir);
            sky_sample(dir, &img[i*3]); ++n_esc;
        }
        else { img[i*3]=255; img[i*3+1]=0; img[i*3+2]=255; ++n_div; }
    }

    printf("kernel: %.3f ms for %d rays  =>  %.2f Mray/s\n",
           ms, n_pixels, (double)n_pixels / (ms * 1e3));
    printf("captured=%d  escaped=%d  diverged=%d\n", n_cap, n_esc, n_div);

    const char* out = "sch_gpu.ppm";
    FILE* fp = fopen(out, "wb");
    if (!fp) { perror("open"); return 1; }
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(img, 1, n_pixels * 3, fp);
    fclose(fp);
    printf("wrote %s\n", out);

    delete[] term; delete[] final_state; delete[] img;
    cudaFree(d_term); cudaFree(d_final);
    return 0;
}
