//! Numerical curvature: Riemann, Ricci, Einstein tensor, stress-energy.
//!
//! Built generically on top of `Metric::christoffel`. Second derivatives of
//! the metric (needed for Riemann) come from finite differences of the
//! Christoffel symbols, not finite differences of g.  Doing it this way
//! halves the order of differentiation we apply numerically and gives noise
//! at h² rather than h, which matters because we'll be reading off
//! near-zero energy densities and looking for *sign* changes.
//!
//! Conventions (Misner-Thorne-Wheeler / Wald):
//!
//!   R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
//!   R_μν    = R^α_μαν                              (Ricci)
//!   R       = g^μν R_μν                            (Ricci scalar)
//!   G_μν    = R_μν − ½ g_μν R                     (Einstein)
//!   T_μν    = G_μν / (8π)                          (units G = c = 1)

use crate::metric::Metric;
use crate::types::{Christoffel, Mat4, Scalar, Vec4};

/// Riemann tensor with the first index raised: R^ρ_σμν as `r[rho][sigma][mu][nu]`.
pub type Riemann = [[[[Scalar; 4]; 4]; 4]; 4];

/// All ∂_α Γ^μ_βγ as `dgamma[alpha][mu][beta][gamma]`.
fn d_christoffel<M: Metric>(metric: &M, x: &Vec4) -> [Christoffel; 4] {
    let h = metric.fd_step(x);
    let mut out = [[[[0.0; 4]; 4]; 4]; 4];
    for alpha in 0..4 {
        let mut xp = *x;
        let mut xm = *x;
        xp[alpha] += h[alpha];
        xm[alpha] -= h[alpha];
        let gp = metric.christoffel(&xp);
        let gm = metric.christoffel(&xm);
        let inv_2h = 1.0 / (2.0 * h[alpha]);
        for mu in 0..4 {
            for beta in 0..4 {
                for gamma in 0..4 {
                    out[alpha][mu][beta][gamma] =
                        (gp[mu][beta][gamma] - gm[mu][beta][gamma]) * inv_2h;
                }
            }
        }
    }
    out
}

pub fn riemann<M: Metric>(metric: &M, x: &Vec4) -> Riemann {
    let gamma = metric.christoffel(x);
    let dgamma = d_christoffel(metric, x);
    let mut r: Riemann = [[[[0.0; 4]; 4]; 4]; 4];
    for rho in 0..4 {
        for sigma in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    let term1 = dgamma[mu][rho][nu][sigma];
                    let term2 = dgamma[nu][rho][mu][sigma];
                    let mut term3 = 0.0;
                    let mut term4 = 0.0;
                    for lambda in 0..4 {
                        term3 += gamma[rho][mu][lambda] * gamma[lambda][nu][sigma];
                        term4 += gamma[rho][nu][lambda] * gamma[lambda][mu][sigma];
                    }
                    r[rho][sigma][mu][nu] = term1 - term2 + term3 - term4;
                }
            }
        }
    }
    r
}

pub fn ricci<M: Metric>(metric: &M, x: &Vec4) -> Mat4 {
    let r = riemann(metric, x);
    let mut ric = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            let mut s = 0.0;
            for alpha in 0..4 {
                s += r[alpha][mu][alpha][nu];
            }
            ric[mu][nu] = s;
        }
    }
    ric
}

pub fn ricci_scalar<M: Metric>(metric: &M, x: &Vec4) -> Scalar {
    let g_inv = metric.g_inv(x);
    let ric = ricci(metric, x);
    let mut s = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            s += g_inv[mu][nu] * ric[mu][nu];
        }
    }
    s
}

pub fn einstein<M: Metric>(metric: &M, x: &Vec4) -> Mat4 {
    let g = metric.g(x);
    let ric = ricci(metric, x);
    let r = ricci_scalar(metric, x);
    let mut ein = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            ein[mu][nu] = ric[mu][nu] - 0.5 * g[mu][nu] * r;
        }
    }
    ein
}

pub fn stress_energy<M: Metric>(metric: &M, x: &Vec4) -> Mat4 {
    let ein = einstein(metric, x);
    let factor = 1.0 / (8.0 * std::f64::consts::PI);
    let mut t = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            t[mu][nu] = ein[mu][nu] * factor;
        }
    }
    t
}

/// Null Energy Condition contraction: T_μν k^μ k^ν for a null vector k. NEC
/// holds iff this is ≥ 0 for *every* null k. We return the value, the caller
/// decides what to do with negatives.
pub fn nec_contraction<M: Metric>(metric: &M, x: &Vec4, k: &Vec4) -> Scalar {
    let t = stress_energy(metric, x);
    let mut s = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            s += t[mu][nu] * k[mu] * k[nu];
        }
    }
    s
}

/// Weak Energy Condition contraction: T_μν u^μ u^ν for a timelike u.
pub fn wec_contraction<M: Metric>(metric: &M, x: &Vec4, u: &Vec4) -> Scalar {
    nec_contraction(metric, x, u)
}
