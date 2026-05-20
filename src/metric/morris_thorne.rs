//! Morris–Thorne / Ellis traversable wormhole metric. Phase 2.
//!
//!   ds² = -dt² + dl² + (b₀² + l²)(dθ² + sin²θ dφ²)
//!
//! `l` is the proper radial coordinate; l < 0 and l > 0 are the two
//! asymptotic regions joined at the throat l = 0 of radius b₀.

use super::{Metric, Termination};
use crate::types::{Mat4, Scalar, Vec4, ZERO_MAT4};

pub struct MorrisThorne {
    pub throat_radius: Scalar,
}

impl Default for MorrisThorne {
    fn default() -> Self {
        Self { throat_radius: 1.0 }
    }
}

impl Metric for MorrisThorne {
    fn g(&self, x: &Vec4) -> Mat4 {
        // Coordinates: (t, l, θ, φ).
        let l = x[1];
        let theta = x[2];
        let b0 = self.throat_radius;
        let r2 = b0 * b0 + l * l;
        let sin_t = theta.sin();

        let mut g = ZERO_MAT4;
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = r2;
        g[3][3] = r2 * sin_t * sin_t;
        g
    }

    fn terminate(&self, x: &Vec4) -> Option<Termination> {
        let l = x[1];
        if !l.is_finite() {
            return Some(Termination::Diverged);
        }
        if l.abs() > 1000.0 {
            Some(Termination::Escaped)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "morris_thorne"
    }

    fn asymptotic_direction(&self, x: &Vec4, p: &Vec4) -> [Scalar; 3] {
        // Coordinates (t, l, θ, φ). At |l| → ∞ both asymptotic regions are
        // flat; the embedding radius is |l|. The sign of l selects which
        // universe the ray escaped to — we encode it in vx for visualisation
        // ("through the wormhole" vs "around it").
        let l = x[1];
        let theta = x[2];
        let phi = x[3];
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let sin_p = phi.sin();
        let cos_p = phi.cos();
        let pl = p[1];
        let pt = p[2];
        let pp = p[3];
        let r_eff = l.abs().max(1e-30);

        let vx = sin_t * cos_p * pl + r_eff * cos_t * cos_p * pt - r_eff * sin_t * sin_p * pp;
        let vy = sin_t * sin_p * pl + r_eff * cos_t * sin_p * pt + r_eff * sin_t * cos_p * pp;
        let vz = cos_t * pl - r_eff * sin_t * pt;
        let n = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-30);
        let sign = if l < 0.0 { -1.0 } else { 1.0 };
        [sign * vx / n, vy / n, vz / n]
    }
}
