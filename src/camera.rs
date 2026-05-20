//! Camera and null-ray initial conditions.
//!
//! We build a local orthonormal tetrad at the observer (Gram–Schmidt
//! against the metric), pick a "forward" axis pointing at the origin of the
//! coordinate chart, generate per-pixel directions in that frame, and lift
//! them to the coordinate basis via the tetrad.
//!
//! Doing this generically (rather than hardcoding the Schwarzschild tetrad)
//! is what lets the same camera code work for Alcubierre and wormhole
//! metrics in Phase 2.

use crate::integrator::State;
use crate::metric::Metric;
use crate::types::{Mat4, Scalar, Vec4, dot4};

/// Spherical-style observer at fixed coordinates. The "forward" image axis
/// is built from the observer's spatial radial direction (toward decreasing
/// `r` or its analogue). This is the right framing for spherically
/// symmetric metrics in (t, r, θ, φ) charts and the Cartesian Alcubierre
/// chart alike — the convention is just "look at coordinate origin."
pub struct Camera {
    pub observer: Vec4,
    pub fov_y_rad: Scalar,
    pub width: usize,
    pub height: usize,
}

impl Camera {
    pub fn new(observer: Vec4, fov_y_deg: Scalar, width: usize, height: usize) -> Self {
        Self {
            observer,
            fov_y_rad: fov_y_deg.to_radians(),
            width,
            height,
        }
    }

    /// Build the observer's local orthonormal tetrad (e₀, e₁, e₂, e₃)
    /// against the metric at the observer position. Returns four 4-vectors
    /// stored in the rows of a Mat4: tetrad[a] = e_(a)^μ.
    ///
    /// Strategy:
    ///  - e₀: unit timelike. For a static observer in a stationary chart,
    ///    e₀^μ ∝ δ^μ_0; we just normalize it.
    ///  - e₁, e₂, e₃: start from coordinate spatial directions δ^μ_i and
    ///    Gram–Schmidt against the metric.
    pub fn tetrad<M: Metric>(&self, metric: &M) -> Mat4 {
        let g = metric.g(&self.observer);
        let mut tet = [[0.0; 4]; 4];

        // e₀ = δ^μ_0 / √(-g_tt)
        let gtt = g[0][0];
        assert!(gtt < 0.0, "observer is not in a timelike region (g_tt >= 0)");
        let n0 = 1.0 / (-gtt).sqrt();
        tet[0][0] = n0;

        // Spatial seeds: index by coordinate. Order chosen so e₁ is the
        // "radial-like" direction (chart index 1) — that's what the image
        // forward axis is built on.
        let seeds: [Vec4; 3] = [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..3 {
            let mut v = seeds[i];
            // Subtract projections onto previously built e₀..e_i.
            for k in 0..=i {
                let ek = tet[k];
                // ⟨e_k, v⟩ = g(e_k, v); the sign of g(e_k, e_k) matters.
                let proj = dot4(&g, &ek, &v);
                let norm_k = dot4(&g, &ek, &ek);
                // For timelike e₀: g(e₀,e₀) = -1, so the projection coefficient
                // is proj / (-1) = -proj. For spacelike e_k (k>=1): coefficient
                // is proj / +1 = proj. The generic formula proj / g(e_k,e_k)
                // covers both.
                let coeff = proj / norm_k;
                for mu in 0..4 {
                    v[mu] -= coeff * ek[mu];
                }
            }
            // Normalise to unit spacelike.
            let n2 = dot4(&g, &v, &v);
            assert!(
                n2 > 0.0,
                "Gram–Schmidt produced non-spacelike vector (n² = {n2}) — chart broken at this point"
            );
            let n = n2.sqrt();
            for mu in 0..4 {
                v[mu] /= n;
            }
            tet[i + 1] = v;
        }

        tet
    }

    /// Generate the initial state for the ray that the pixel (i, j) "looks
    /// along." Convention:
    ///   - image forward = -e₁  (toward decreasing chart coord 1)
    ///   - image right   = +e₃
    ///   - image up      = -e₂
    ///
    /// `tetrad` is the matrix built by `Self::tetrad`. We pass it in rather
    /// than recompute per pixel.
    pub fn ray_state(&self, tetrad: &Mat4, i: usize, j: usize) -> State {
        self.ray_state_sub(tetrad, i, j, 0.5, 0.5)
    }

    /// As `ray_state`, but with explicit sub-pixel offsets in [0, 1]².
    /// Used by supersampling to anti-alias the procedural sky.
    pub fn ray_state_sub(
        &self,
        tetrad: &Mat4,
        i: usize,
        j: usize,
        du: Scalar,
        dv: Scalar,
    ) -> State {
        let aspect = self.width as Scalar / self.height as Scalar;
        let h = (self.fov_y_rad * 0.5).tan();
        let w = h * aspect;

        // [-1,1] → image plane; flip y so screen-y points down.
        let u = (2.0 * (i as Scalar + du) / self.width as Scalar - 1.0) * w;
        let v = -(2.0 * (j as Scalar + dv) / self.height as Scalar - 1.0) * h;

        // Direction in local frame: (n₁, n₂, n₃).
        // forward = (-1, 0, 0); right = (0, 0, +1); up = (0, -1, 0).
        let dx = -1.0;
        let dy = -v;
        let dz = u;
        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
        let n1 = dx / norm;
        let n2 = dy / norm;
        let n3 = dz / norm;

        // Coordinate-basis 4-momentum: p^μ = e₀^μ + n_i e_(i+1)^μ.
        // (We rescale by an arbitrary positive factor; only the direction
        // matters for null geodesics, and we keep p^0 ≈ +1 for sanity.)
        let mut p = [0.0; 4];
        for mu in 0..4 {
            p[mu] = tetrad[0][mu]
                + n1 * tetrad[1][mu]
                + n2 * tetrad[2][mu]
                + n3 * tetrad[3][mu];
        }

        [
            self.observer[0],
            self.observer[1],
            self.observer[2],
            self.observer[3],
            p[0],
            p[1],
            p[2],
            p[3],
        ]
    }
}
