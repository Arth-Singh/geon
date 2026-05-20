//! Schwarzschild metric in standard (t, r, θ, φ) coordinates.
//!
//! ds² = -(1 - 2M/r) dt² + (1 - 2M/r)^(-1) dr² + r² dθ² + r² sin²θ dφ²
//!
//! We carry analytic Christoffel symbols because Phase 1 hinges on this
//! metric reproducing the known shadow radius 3√3 M; any error here
//! contaminates Phase 2 and Phase 3.

use super::{Metric, Termination};
use crate::types::{Christoffel, Mat4, Scalar, Vec4, ZERO_MAT4};

pub struct Schwarzschild {
    pub mass: Scalar,
    /// Coordinate `r` at which to treat the photon as captured. Slightly
    /// outside the horizon avoids the metric blowing up.
    pub horizon_pad: Scalar,
    /// Coordinate `r` past which the photon has escaped. Should be much
    /// larger than the impact parameters of interest.
    pub escape_radius: Scalar,
}

impl Schwarzschild {
    pub fn new(mass: Scalar) -> Self {
        Self {
            mass,
            horizon_pad: 1.001,
            escape_radius: 1000.0,
        }
    }
}

impl Metric for Schwarzschild {
    fn g(&self, x: &Vec4) -> Mat4 {
        let r = x[1];
        let theta = x[2];
        let m = self.mass;
        let f = 1.0 - 2.0 * m / r;
        let sin_t = theta.sin();
        let mut g = ZERO_MAT4;
        g[0][0] = -f;
        g[1][1] = 1.0 / f;
        g[2][2] = r * r;
        g[3][3] = r * r * sin_t * sin_t;
        g
    }

    fn g_inv(&self, x: &Vec4) -> Mat4 {
        let r = x[1];
        let theta = x[2];
        let m = self.mass;
        let f = 1.0 - 2.0 * m / r;
        let sin_t = theta.sin();
        let mut g = ZERO_MAT4;
        g[0][0] = -1.0 / f;
        g[1][1] = f;
        g[2][2] = 1.0 / (r * r);
        g[3][3] = 1.0 / (r * r * sin_t * sin_t);
        g
    }

    fn christoffel(&self, x: &Vec4) -> Christoffel {
        let r = x[1];
        let theta = x[2];
        let m = self.mass;
        let f = 1.0 - 2.0 * m / r;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        let mut g = [[[0.0; 4]; 4]; 4];

        // Γ^t_{t r} = Γ^t_{r t}
        let gtr = m / (r * r * f);
        g[0][0][1] = gtr;
        g[0][1][0] = gtr;

        // Γ^r_{t t}
        g[1][0][0] = m * f / (r * r);
        // Γ^r_{r r}
        g[1][1][1] = -m / (r * r * f);
        // Γ^r_{θ θ}
        g[1][2][2] = -(r - 2.0 * m);
        // Γ^r_{φ φ}
        g[1][3][3] = -(r - 2.0 * m) * sin_t * sin_t;

        // Γ^θ_{r θ} = Γ^θ_{θ r}
        let gtheta_rt = 1.0 / r;
        g[2][1][2] = gtheta_rt;
        g[2][2][1] = gtheta_rt;
        // Γ^θ_{φ φ}
        g[2][3][3] = -sin_t * cos_t;

        // Γ^φ_{r φ} = Γ^φ_{φ r}
        g[3][1][3] = 1.0 / r;
        g[3][3][1] = 1.0 / r;
        // Γ^φ_{θ φ} = Γ^φ_{φ θ}
        // cot θ — guard against poles. We don't render rays that approach θ=0
        // or θ=π closely, but defensively clamp here.
        let cot = if sin_t.abs() < 1e-8 {
            0.0
        } else {
            cos_t / sin_t
        };
        g[3][2][3] = cot;
        g[3][3][2] = cot;

        g
    }

    fn terminate(&self, x: &Vec4) -> Option<Termination> {
        let r = x[1];
        if !r.is_finite() {
            return Some(Termination::Diverged);
        }
        if r < self.horizon_pad * 2.0 * self.mass {
            Some(Termination::Captured)
        } else if r > self.escape_radius * self.mass {
            Some(Termination::Escaped)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "schwarzschild"
    }
}
