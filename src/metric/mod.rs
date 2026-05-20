//! The `Metric` abstraction: anything we can ray-trace through.
//!
//! Each concrete metric only has to supply g(x) and ∂g(x). The Christoffel
//! symbols and the inverse metric are derived generically, so adding a new
//! spacetime is a few lines of math.
//!
//! For metrics with closed-form Christoffels (Schwarzschild, Minkowski) the
//! implementation can override `christoffel` to skip the finite-difference
//! cost — useful for Phase 1 correctness checks where we want as little
//! numerical noise as possible.

use crate::types::{Christoffel, Mat4, Scalar, Vec4, inv4};

pub mod alcubierre;
pub mod minkowski;
pub mod morris_thorne;
pub mod schwarzschild;

pub trait Metric: Sync + Send {
    /// g_μν at coordinate x.
    fn g(&self, x: &Vec4) -> Mat4;

    /// ∂_α g_μν at coordinate x. Default: central finite difference on `g`.
    /// Override for analytic metrics where this matters for accuracy.
    fn dg(&self, x: &Vec4) -> [Mat4; 4] {
        let h = self.fd_step(x);
        let mut out = [[[0.0; 4]; 4]; 4];
        for alpha in 0..4 {
            let mut xp = *x;
            let mut xm = *x;
            xp[alpha] += h[alpha];
            xm[alpha] -= h[alpha];
            let gp = self.g(&xp);
            let gm = self.g(&xm);
            let two_h = 2.0 * h[alpha];
            for mu in 0..4 {
                for nu in 0..4 {
                    out[alpha][mu][nu] = (gp[mu][nu] - gm[mu][nu]) / two_h;
                }
            }
        }
        out
    }

    /// Inverse metric g^μν. Default: numerical inversion of `g`.
    fn g_inv(&self, x: &Vec4) -> Mat4 {
        inv4(&self.g(x))
    }

    /// Christoffel symbols Γ^μ_αβ = ½ g^μσ (∂_α g_σβ + ∂_β g_σα - ∂_σ g_αβ).
    fn christoffel(&self, x: &Vec4) -> Christoffel {
        let ginv = self.g_inv(x);
        let dg = self.dg(x);
        let mut gamma = [[[0.0; 4]; 4]; 4];
        for mu in 0..4 {
            for alpha in 0..4 {
                for beta in 0..4 {
                    let mut s = 0.0;
                    for sigma in 0..4 {
                        s += ginv[mu][sigma]
                            * (dg[alpha][sigma][beta] + dg[beta][sigma][alpha]
                                - dg[sigma][alpha][beta]);
                    }
                    gamma[mu][alpha][beta] = 0.5 * s;
                }
            }
        }
        gamma
    }

    /// Per-coordinate finite-difference step. Override to scale with local
    /// length scales (e.g., `r` near a horizon).
    fn fd_step(&self, _x: &Vec4) -> [Scalar; 4] {
        [1e-5; 4]
    }

    /// Should the integrator stop here? `Some(reason)` halts the geodesic.
    fn terminate(&self, x: &Vec4) -> Option<Termination> {
        let _ = x;
        None
    }

    /// Human-readable name for logs and image filenames.
    fn name(&self) -> &'static str;

    /// Given a final state (position + 4-momentum) where the ray has
    /// `Escaped`, return the Cartesian direction on the celestial sphere
    /// from which the photon came. Default assumes a spherical chart
    /// (t, r, θ, φ) — override for Cartesian charts (Alcubierre) and
    /// double-sheeted charts (Morris–Thorne with l < 0 sky).
    fn asymptotic_direction(&self, x: &Vec4, p: &Vec4) -> [Scalar; 3] {
        let r = x[1];
        let theta = x[2];
        let phi = x[3];
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let sin_p = phi.sin();
        let cos_p = phi.cos();
        let pr = p[1];
        let pt = p[2];
        let pp = p[3];

        let vx = sin_t * cos_p * pr + r * cos_t * cos_p * pt - r * sin_t * sin_p * pp;
        let vy = sin_t * sin_p * pr + r * cos_t * sin_p * pt + r * sin_t * cos_p * pp;
        let vz = cos_t * pr - r * sin_t * pt;
        let n = (vx * vx + vy * vy + vz * vz).sqrt().max(1e-30);
        [vx / n, vy / n, vz / n]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Geodesic crossed a horizon or fell into a singularity. Render as black.
    Captured,
    /// Geodesic escaped to (numerical) infinity. Render with sky texture.
    Escaped,
    /// Something pathological — NaN, step rejected too many times. Render as
    /// a debug colour so we see it on screen instead of silently lying.
    Diverged,
}
