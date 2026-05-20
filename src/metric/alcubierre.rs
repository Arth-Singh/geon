//! Alcubierre warp drive metric. Phase 2 target — stubbed here so the module
//! tree compiles. Full implementation lands once Schwarzschild validates.
//!
//! In 3+1 form (Alcubierre 1994):
//!   ds² = -dt² + (dx - v_s(t) f(r_s) dt)² + dy² + dz²
//! where r_s is the distance from the bubble centre and f(r_s) is a smooth
//! top-hat. The stress-energy required violates the weak energy condition —
//! quantifying that violation across (bubble radius, v_s) is the real Phase 2
//! payoff.

use super::{Metric, Termination};
use crate::types::{Mat4, Scalar, Vec4, ZERO_MAT4};

pub struct Alcubierre {
    pub bubble_radius: Scalar,
    pub bubble_thickness: Scalar,
    pub ship_velocity: Scalar,
    pub ship_x: Scalar,
}

impl Default for Alcubierre {
    fn default() -> Self {
        Self {
            bubble_radius: 2.0,
            bubble_thickness: 0.5,
            ship_velocity: 2.0,
            ship_x: 0.0,
        }
    }
}

impl Alcubierre {
    fn shape(&self, x: &Vec4) -> Scalar {
        let dx = x[1] - self.ship_x;
        let dy = x[2];
        let dz = x[3];
        let r_s = (dx * dx + dy * dy + dz * dz).sqrt();
        let s = self.bubble_thickness;
        let r = self.bubble_radius;
        let t_plus = ((r_s + r) / s).tanh();
        let t_minus = ((r_s - r) / s).tanh();
        let denom = 2.0 * (r / s).tanh();
        (t_plus - t_minus) / denom
    }
}

impl Metric for Alcubierre {
    fn g(&self, x: &Vec4) -> Mat4 {
        let v = self.ship_velocity;
        let f = self.shape(x);
        let mut g = ZERO_MAT4;
        // Cartesian (t, x, y, z), signature (-,+,+,+):
        //   g_tt = -1 + v² f²
        //   g_tx = g_xt = -v f
        //   g_xx = g_yy = g_zz = 1
        g[0][0] = -1.0 + v * v * f * f;
        g[0][1] = -v * f;
        g[1][0] = -v * f;
        g[1][1] = 1.0;
        g[2][2] = 1.0;
        g[3][3] = 1.0;
        g
    }

    fn terminate(&self, x: &Vec4) -> Option<Termination> {
        let r2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        if !r2.is_finite() {
            return Some(Termination::Diverged);
        }
        if r2 > 1e6 {
            Some(Termination::Escaped)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "alcubierre"
    }

    fn asymptotic_direction(&self, _x: &Vec4, p: &Vec4) -> [Scalar; 3] {
        // Cartesian chart — spatial momentum *is* the direction.
        let vx = p[1];
        let vy = p[2];
        let vz = p[3];
        let n = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-30);
        [vx / n, vy / n, vz / n]
    }
}
