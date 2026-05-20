//! Core spacetime types.
//!
//! Convention: signature (-, +, +, +). Indices run 0..=3. Index 0 is the
//! "time-like" coordinate for every metric we currently care about, even when
//! the coordinate chart twists that interpretation locally (Alcubierre).

pub type Scalar = f64;

/// Four-vector. Stored contravariantly (upper index) by default.
pub type Vec4 = [Scalar; 4];

/// Rank-2 spacetime tensor. `m[i][j]` is the (i, j) component.
pub type Mat4 = [[Scalar; 4]; 4];

/// Christoffel symbols Γ^μ_αβ: `c[mu][alpha][beta]`.
pub type Christoffel = [[[Scalar; 4]; 4]; 4];

pub const ZERO_VEC4: Vec4 = [0.0; 4];
pub const ZERO_MAT4: Mat4 = [[0.0; 4]; 4];

#[inline(always)]
pub fn dot4(g: &Mat4, a: &Vec4, b: &Vec4) -> Scalar {
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            s += g[i][j] * a[i] * b[j];
        }
    }
    s
}

/// Invert a 4x4 symmetric matrix. Uses nalgebra for robustness near coordinate
/// singularities (we'll hit them — Schwarzschild horizon, polar axis).
pub fn inv4(m: &Mat4) -> Mat4 {
    use nalgebra::Matrix4;
    let nm = Matrix4::from_fn(|i, j| m[i][j]);
    let inv = nm
        .try_inverse()
        .expect("metric became singular — coordinate chart broke down");
    let mut out = ZERO_MAT4;
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = inv[(i, j)];
        }
    }
    out
}
