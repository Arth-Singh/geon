//! Celestial sphere texture. The point is not photorealism — we want
//! gravitational lensing to be *visible*, which means high-frequency,
//! recognisable features around the shadow.
//!
//! Strategy: a coloured 6-sector pattern (so we can tell which hemisphere
//! a deflected ray came from) plus a procedural hashed starfield (so we
//! see Einstein rings and multiple-image artefacts).
//!
//! Texture is rotated by `PHI_OFFSET` so that no checker boundary aligns
//! with phi=π (the natural look-at direction). Otherwise the seam between
//! adjacent checker cells looks like a "wrapping bug" to a casual reader,
//! when it's really just unfortunate alignment.

use crate::types::Scalar;

const PHI_OFFSET: Scalar = 0.37;
const THETA_OFFSET: Scalar = 0.21;

/// Toggle the procedural starfield. With stars on we get Einstein-ring
/// speckle effects that prove the photon-sphere whirling is real; with
/// stars off we get a smooth checker that makes the secondary-image
/// structure obvious as a coherent distortion rather than noise.
pub static SHOW_STARS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[inline]
fn hash1(x: u64) -> u64 {
    let mut x = x.wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

#[inline]
fn hash2(a: i64, b: i64) -> u64 {
    hash1((a as u64).wrapping_mul(73856093) ^ (b as u64).wrapping_mul(19349663))
}

/// Procedural star field. Subdivides the (θ, φ) plane into cells; each cell
/// has a small chance of containing a point-like star.
fn stars(theta: Scalar, phi: Scalar) -> [f32; 3] {
    if !SHOW_STARS.load(std::sync::atomic::Ordering::Relaxed) {
        return [0.0, 0.0, 0.0];
    }
    let n_phi = 512.0;
    let n_theta = 256.0;
    let phi_shift = (phi + PHI_OFFSET).rem_euclid(std::f64::consts::TAU);
    let theta_shift = (theta + THETA_OFFSET).clamp(0.0, std::f64::consts::PI);
    let i = (phi_shift / std::f64::consts::TAU * n_phi).floor() as i64;
    let j = (theta_shift / std::f64::consts::PI * n_theta).floor() as i64;

    let h = hash2(i, j);
    let prob = (h & 0xFFFF) as f64 / 65535.0;

    if prob < 0.004 {
        let brightness = 0.4 + 0.6 * (((h >> 16) & 0xFF) as f64 / 255.0);
        let temp = ((h >> 24) & 0xFF) as f64 / 255.0;
        // Star colour: tint toward blue for hot, red for cool.
        let r = (brightness * (1.0 - 0.3 * temp)) as f32;
        let g = (brightness * (1.0 - 0.1 * (temp - 0.5).abs())) as f32;
        let b = (brightness * (0.7 + 0.3 * temp)) as f32;
        [r, g, b]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// 6-sector colour pattern in (θ, φ). Bright enough to win against the deep
/// black of the BH shadow, dim enough not to wash out the stars.
fn sectors(theta: Scalar, phi: Scalar) -> [f32; 3] {
    let phi_shift = (phi + PHI_OFFSET).rem_euclid(std::f64::consts::TAU);
    let theta_shift = (theta + THETA_OFFSET).clamp(0.0, std::f64::consts::PI);
    let phi_mod = (phi_shift / std::f64::consts::TAU * 6.0).floor() as i64;
    let band = (theta_shift / std::f64::consts::PI * 3.0).floor() as i64;

    // Coarse colour grid.
    let palette = [
        [0.15, 0.08, 0.20],
        [0.08, 0.15, 0.22],
        [0.20, 0.15, 0.08],
        [0.10, 0.18, 0.10],
        [0.22, 0.10, 0.10],
        [0.10, 0.10, 0.22],
    ];
    let base = palette[(phi_mod.rem_euclid(6)) as usize];
    let band_mod = if band.rem_euclid(2) == 0 { 1.0 } else { 0.6 };

    // Checkerboard overlay so distortion is obvious. The offsets above keep
    // its cell boundaries off the camera-aligned axes (otherwise the centre
    // of every frame looks like it has a vertical seam, which it doesn't).
    let nphi = 24;
    let ntheta = 12;
    let ic = (phi_shift / std::f64::consts::TAU * nphi as f64).floor() as i64;
    let jc = (theta_shift / std::f64::consts::PI * ntheta as f64).floor() as i64;
    let check = if (ic + jc).rem_euclid(2) == 0 { 1.0 } else { 0.55 };

    [
        (base[0] * band_mod * check) as f32,
        (base[1] * band_mod * check) as f32,
        (base[2] * band_mod * check) as f32,
    ]
}

/// Sample the celestial sphere along a unit Cartesian direction.
pub fn sample(dir_cart: [Scalar; 3]) -> [f32; 3] {
    let [x, y, z] = dir_cart;
    let r = (x * x + y * y + z * z).sqrt().max(1e-30);
    let theta = (z / r).clamp(-1.0, 1.0).acos();
    let phi = y.atan2(x);

    let s = sectors(theta, phi);
    let st = stars(theta, phi);
    [
        (s[0] + st[0]).min(1.0),
        (s[1] + st[1]).min(1.0),
        (s[2] + st[2]).min(1.0),
    ]
}
